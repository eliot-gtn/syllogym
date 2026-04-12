#!/usr/bin/env python3
"""
enrich_episodes.py
------------------
Offline one-shot script: sample episodes from all SylloGym generators,
enrich with Claude Haiku, verify, and save to JSON banks.

Resilience features:
  - Checkpoints: each episode is written to disk immediately after enrichment.
    If the script crashes, re-running resumes from the last completed episode.
  - Exponential backoff: rate limit / server errors are retried up to 5 times.
  - Generator-level resume: already-completed generators are skipped on re-run.

Usage:
    # Key loaded automatically from .env — just run:
    python scripts/enrich_episodes.py --output-dir enriched/ --n-episodes 500

    # Dry run (no API calls, pipeline test only):
    python scripts/enrich_episodes.py --dry-run --n-episodes 10

    # Single generator:
    python scripts/enrich_episodes.py --generator diversity --n-episodes 500

    # Resume after crash — re-run the same command, already-done episodes are skipped.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

# Add project roots to path
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "envs"))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))

from enricher.enricher import EpisodeEnricher, _episode_to_dict

# Import all generators
from syllogym_env.server.generators.diversity_generator import DiversityGenerator
from syllogym_env.server.generators.ucc_generator import UCCGenerator
from syllogym_env.server.generators.miranda_generator import MirandaGenerator
from syllogym_env.server.generators.consideration_generator import ConsiderationGenerator
from syllogym_env.server.generators.mens_rea_generator import MensReaGenerator
from syllogym_env.server.generators.terry_stop_generator import TerryStopGenerator
from syllogym_env.server.generators.sara_driver import SARADriver
from syllogym_env.server.generators.tsr_generator import TSRGenerator
from syllogym_env.server.generators.qualifying_child_generator import QualifyingChildGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("enrich_episodes")


# ---------------------------------------------------------------------------
# Generator registry — order matters for resume (stable order)
# ---------------------------------------------------------------------------

ALL_GENERATORS: dict[str, tuple] = {
    "diversity":     (DiversityGenerator(),         None),   # mixed mode
    "ucc":           (UCCGenerator(),               None),
    "miranda":       (MirandaGenerator(),           None),
    "consideration": (ConsiderationGenerator(),     None),
    "mens_rea":      (MensReaGenerator(),           None),
    "terry":         (TerryStopGenerator(),         None),
    "sara":          (SARADriver(),                 None),
    "tsr":           (TSRGenerator(),               None),
    "qc":            (QualifyingChildGenerator(),   None),   # handles qc_* and qr_*
}


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _checkpoint_path(output_dir: Path, name: str) -> Path:
    """Path to the JSONL checkpoint file for a generator (one episode per line)."""
    return output_dir / f"{name}_enriched.jsonl"


def _final_path(output_dir: Path, name: str) -> Path:
    """Path to the final JSON array output for a generator."""
    return output_dir / f"{name}_enriched.json"


def _load_checkpoint(ckpt_path: Path) -> list[dict]:
    """Load already-completed episodes from a JSONL checkpoint."""
    if not ckpt_path.exists():
        return []
    episodes = []
    with open(ckpt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    episodes.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed checkpoint line in %s", ckpt_path)
    return episodes


def _append_checkpoint(ckpt_path: Path, episode_dict: dict) -> None:
    """Append one episode to the JSONL checkpoint (atomic per-line write)."""
    with open(ckpt_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(episode_dict, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _finalize_checkpoint(ckpt_path: Path, final_path: Path) -> None:
    """Convert JSONL checkpoint → final JSON array and remove checkpoint."""
    episodes = _load_checkpoint(ckpt_path)
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(episodes, f, indent=2, ensure_ascii=False)
    ckpt_path.unlink()
    logger.info("Finalized %d episodes → %s", len(episodes), final_path)


# ---------------------------------------------------------------------------
# Per-generator enrichment with checkpointing
# ---------------------------------------------------------------------------

def enrich_generator_with_checkpoints(
    name: str,
    generator,
    enricher: EpisodeEnricher,
    output_dir: Path,
    n_episodes: int,
    seed: int,
) -> int:
    """
    Enrich `n_episodes` from `generator`, writing each to a JSONL checkpoint.
    Resumes automatically from the last completed episode on re-run.
    Returns the number of episodes enriched in this run (0 if fully resumed).
    """
    import random

    ckpt_path = _checkpoint_path(output_dir, name)
    final_path = _final_path(output_dir, name)

    # If final JSON already exists and has enough episodes → skip entirely
    if final_path.exists():
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if len(existing) >= n_episodes:
            logger.info("=== %s: already complete (%d episodes), skipping ===", name, len(existing))
            return 0

    # Load checkpoint (partial progress from a previous run)
    done_episodes = _load_checkpoint(ckpt_path)
    done_seeds = {ep["seed"] for ep in done_episodes}
    n_done = len(done_episodes)

    if n_done > 0:
        logger.info("=== %s: resuming from episode %d/%d ===", name, n_done, n_episodes)
    else:
        logger.info("=== %s: starting fresh (%d episodes) ===", name, n_episodes)

    n_new = 0
    for i in range(n_episodes):
        ep_seed = seed + i
        if ep_seed in done_seeds:
            continue  # already done

        rng = random.Random(ep_seed)
        ep = generator.sample(rng, task_name=None)
        if ep is None:
            logger.warning("Generator %s returned None for seed=%d", name, ep_seed)
            continue

        enriched = enricher.enrich_episode(ep, seed=ep_seed)
        ep_dict = _episode_to_dict(enriched)
        _append_checkpoint(ckpt_path, ep_dict)
        n_new += 1

        # Progress log every 50 episodes
        total_done = n_done + n_new
        if total_done % 50 == 0:
            stats = enricher.stats()
            t = stats["total_fields"]
            e = stats["enriched"]
            rate = 100 * e // t if t else 0
            logger.info(
                "%s: %d/%d episodes — enrichment rate %d%% | "
                "%d rejected | %d API errors",
                name, total_done, n_episodes, rate,
                stats["rejected_verification"], stats["rejected_api_error"],
            )

    # Finalize: JSONL → JSON array
    _finalize_checkpoint(ckpt_path, final_path)

    stats = enricher.stats()
    t = stats["total_fields"]
    e = stats["enriched"]
    rate = 100 * e // t if t else 0
    logger.info(
        "=== %s done: %d new episodes, enrichment rate %d%% ===",
        name, n_new, rate,
    )
    return n_new


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Offline episode enrichment for SylloGym")
    parser.add_argument(
        "--output-dir", default="enriched",
        help="Directory to write enriched JSON files (default: enriched/)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=500,
        help="Number of episodes per generator (default: 500)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)",
    )
    parser.add_argument(
        "--generator", default=None,
        choices=list(ALL_GENERATORS.keys()),
        help="Run a single generator only",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Skip API calls — test pipeline without spending tokens",
    )
    parser.add_argument(
        "--model", default="claude-haiku-4-5-20251001",
        help="Claude model to use for enrichment",
    )
    args = parser.parse_args()

    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key and not args.dry_run:
        logger.error("ANTHROPIC_API_KEY not set. Use --dry-run for testing.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    enricher = EpisodeEnricher(
        api_key=api_key,
        model=args.model,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        logger.info("DRY RUN — no API calls will be made")

    t0 = time.time()
    total_new = 0

    for name, (generator, _) in ALL_GENERATORS.items():
        if args.generator and name != args.generator:
            continue
        total_new += enrich_generator_with_checkpoints(
            name=name,
            generator=generator,
            enricher=enricher,
            output_dir=output_dir,
            n_episodes=args.n_episodes,
            seed=args.seed,
        )

    elapsed = time.time() - t0
    stats = enricher.stats()
    t = stats["total_fields"]
    e = stats["enriched"]
    logger.info(
        "All done in %.0fs — %d new episodes | %d/%d fields enriched (%d%%) | "
        "%d rejected | %d API errors",
        elapsed, total_new, e, t, 100 * e // t if t else 0,
        stats["rejected_verification"], stats["rejected_api_error"],
    )


if __name__ == "__main__":
    main()
