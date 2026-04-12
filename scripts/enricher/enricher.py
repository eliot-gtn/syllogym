"""
enricher.py
-----------
EpisodeEnricher — generic offline linguistic enrichment for SylloGym episodes.

How it works:
  1. Sample N episodes per task from the generator.
  2. For each episode, send initial_facts + each non-empty new_info to Claude Haiku.
  3. Verify the paraphrase with verifier.verify_paraphrase().
  4. Accept on pass; fall back to original on reject (with detailed logging).
  5. Serialize enriched episodes to a JSON file.

The enriched JSON is consumed at training time as a pre-built bank of episodes.
The environment stays unchanged — enrichment is purely offline.

Usage:
    from scripts.enricher import EpisodeEnricher

    enricher = EpisodeEnricher(api_key="sk-ant-...")
    episodes = enricher.enrich_generator(
        generator=DiversityGenerator(),
        task_name="diversity_3",
        n_episodes=200,
        seed=42,
    )
    enricher.save(episodes, "enriched/diversity.json")
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import anthropic

from .prompt import SYSTEM_PROMPT, build_user_prompt
from .verifier import VerificationResult, verify_paraphrase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enriched episode schema
# ---------------------------------------------------------------------------

@dataclass
class EnrichedTurn:
    new_info: str
    question: str
    correct_answer: str
    valid_answers: list[str]
    is_twist: bool
    # Enrichment metadata
    new_info_enriched: bool = False  # True if new_info was successfully paraphrased


@dataclass
class EnrichedEpisode:
    task_name: str
    rule: str
    initial_facts: str
    initial_facts_enriched: bool    # True if initial_facts was paraphrased
    turns: list[EnrichedTurn]
    difficulty: int
    seed: int


# ---------------------------------------------------------------------------
# Haiku client wrapper
# ---------------------------------------------------------------------------

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 512
_TEMPERATURE = 0.8      # some creativity, but not too wild
_MAX_RETRIES = 5        # per field — enough for transient errors
_RETRY_BASE_DELAY = 2.0 # seconds — doubles each attempt (exponential backoff)
_RETRY_MAX_DELAY = 60.0 # cap at 60s


class _HaikuClient:
    def __init__(self, api_key: str, model: str = _DEFAULT_MODEL) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def paraphrase(self, text: str, field_type: str, task_name: str) -> str | None:
        """
        Call Haiku to paraphrase `text`.
        Returns the paraphrased string, or None after exhausting retries.

        Retry policy (exponential backoff):
          - RateLimitError / OverloadedError: wait and retry up to _MAX_RETRIES times
          - Other APIError: single retry, then give up (transient network issues)
          - Delay: 2s, 4s, 8s, 16s, 32s (capped at 60s)
        """
        user_msg = build_user_prompt(text, field_type, task_name)
        for attempt in range(_MAX_RETRIES + 1):
            try:
                response = self._client.messages.create(
                    model=self._model,
                    max_tokens=_MAX_TOKENS,
                    temperature=_TEMPERATURE,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return response.content[0].text.strip()
            except (anthropic.RateLimitError, anthropic.InternalServerError) as e:
                # Retriable: rate limit or server overload
                if attempt < _MAX_RETRIES:
                    delay = min(_RETRY_BASE_DELAY * (2 ** attempt), _RETRY_MAX_DELAY)
                    logger.warning(
                        "Retriable error (%s) — attempt %d/%d, sleeping %.0fs",
                        type(e).__name__, attempt + 1, _MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                    continue
                logger.warning("Giving up after %d retries: %s", _MAX_RETRIES, e)
                return None
            except anthropic.APIStatusError as e:
                # Non-retriable API errors (bad request, auth, etc.)
                logger.warning("Non-retriable API error %s: %s", e.status_code, e.message)
                return None
            except anthropic.APIConnectionError as e:
                # Network issue — one retry after short delay
                if attempt < 1:
                    logger.warning("Connection error, retrying once: %s", e)
                    time.sleep(5.0)
                    continue
                logger.warning("Connection error after retry: %s", e)
                return None
        return None


# ---------------------------------------------------------------------------
# EpisodeEnricher
# ---------------------------------------------------------------------------

class EpisodeEnricher:
    """
    Generic offline enricher for any SylloGym generator.

    Args:
        api_key:    Anthropic API key.
        model:      Claude model ID (default: claude-haiku-4-5-20251001).
        dry_run:    If True, skip Haiku calls and return originals unchanged.
                    Useful for testing the pipeline without API calls.
    """

    def __init__(
        self,
        api_key: str,
        model: str = _DEFAULT_MODEL,
        dry_run: bool = False,
    ) -> None:
        self._haiku = _HaikuClient(api_key, model) if not dry_run else None
        self._dry_run = dry_run
        self._stats: dict[str, int] = {
            "total_fields": 0,
            "enriched": 0,
            "rejected_verification": 0,
            "rejected_api_error": 0,
        }

    def enrich_episode(
        self,
        episode: Any,   # Episode dataclass from base_generator
        seed: int = 0,
    ) -> EnrichedEpisode:
        """
        Enrich a single Episode. Returns an EnrichedEpisode.

        Fields enriched: `initial_facts` and each non-empty `new_info` in turns.
        All other fields (rule, question, correct_answer) are copied verbatim.
        """
        # Enrich initial_facts
        enriched_initial, initial_enriched = self._enrich_field(
            text=episode.initial_facts,
            field_type="initial_facts",
            task_name=episode.task_name,
        )

        # Enrich each turn's new_info (skip empty)
        enriched_turns: list[EnrichedTurn] = []
        for turn in episode.turns:
            enriched_info = turn.new_info
            info_enriched = False
            if turn.new_info.strip():
                enriched_info, info_enriched = self._enrich_field(
                    text=turn.new_info,
                    field_type="new_info",
                    task_name=episode.task_name,
                )
            enriched_turns.append(EnrichedTurn(
                new_info=enriched_info,
                question=turn.question,
                correct_answer=turn.correct_answer,
                valid_answers=list(turn.valid_answers),
                is_twist=turn.is_twist,
                new_info_enriched=info_enriched,
            ))

        return EnrichedEpisode(
            task_name=episode.task_name,
            rule=episode.rule,
            initial_facts=enriched_initial,
            initial_facts_enriched=initial_enriched,
            turns=enriched_turns,
            difficulty=episode.difficulty,
            seed=seed,
        )

    def enrich_generator(
        self,
        generator: Any,         # BaseGenerator instance
        task_name: str | None,
        n_episodes: int,
        seed: int = 0,
    ) -> list[EnrichedEpisode]:
        """
        Sample and enrich `n_episodes` episodes from `generator`.

        Args:
            generator:   A BaseGenerator instance (DiversityGenerator, etc.)
            task_name:   Specific task name, or None for random from generator.
            n_episodes:  Number of episodes to generate.
            seed:        Base seed (each episode uses seed + i).

        Returns:
            List of EnrichedEpisode objects.
        """
        results: list[EnrichedEpisode] = []

        for i in range(n_episodes):
            ep_seed = seed + i
            rng = random.Random(ep_seed)
            episode = generator.sample(rng, task_name=task_name)
            if episode is None:
                logger.warning("Generator returned None for seed=%d task=%s", ep_seed, task_name)
                continue

            enriched = self.enrich_episode(episode, seed=ep_seed)
            results.append(enriched)

            if (i + 1) % 50 == 0:
                logger.info(
                    "Progress: %d/%d episodes — stats: %s",
                    i + 1, n_episodes, self._stats,
                )

        logger.info("Completed %d episodes — final stats: %s", len(results), self._stats)
        return results

    def save(self, episodes: list[EnrichedEpisode], path: str | Path) -> None:
        """Serialize enriched episodes to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [_episode_to_dict(ep) for ep in episodes]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d enriched episodes to %s", len(episodes), path)

    def stats(self) -> dict[str, int]:
        """Return enrichment statistics."""
        return dict(self._stats)

    def _enrich_field(
        self,
        text: str,
        field_type: str,
        task_name: str,
    ) -> tuple[str, bool]:
        """
        Attempt to enrich a single text field.
        Returns (enriched_text, was_enriched).
        Falls back to original on any failure.
        """
        self._stats["total_fields"] += 1

        if self._dry_run or not text.strip():
            return text, False

        paraphrase = self._haiku.paraphrase(text, field_type, task_name)
        if paraphrase is None:
            self._stats["rejected_api_error"] += 1
            logger.debug("API error — keeping original for %s/%s", task_name, field_type)
            return text, False

        result: VerificationResult = verify_paraphrase(text, paraphrase, task_name)
        if not result.ok:
            self._stats["rejected_verification"] += 1
            logger.debug(
                "Verification failed (%s) for %s/%s — keeping original",
                result.reason, task_name, field_type,
            )
            return text, False

        self._stats["enriched"] += 1
        return paraphrase, True


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _episode_to_dict(ep: EnrichedEpisode) -> dict:
    return {
        "task_name": ep.task_name,
        "rule": ep.rule,
        "initial_facts": ep.initial_facts,
        "initial_facts_enriched": ep.initial_facts_enriched,
        "turns": [
            {
                "new_info": t.new_info,
                "question": t.question,
                "correct_answer": t.correct_answer,
                "valid_answers": t.valid_answers,
                "is_twist": t.is_twist,
                "new_info_enriched": t.new_info_enriched,
            }
            for t in ep.turns
        ],
        "difficulty": ep.difficulty,
        "seed": ep.seed,
    }


def load_enriched_episodes(path: str | Path) -> list[dict]:
    """Load previously enriched episodes from JSON."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)
