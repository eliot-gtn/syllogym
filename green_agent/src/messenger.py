"""
messenger.py
------------
A2A inter-agent communication helper.
Wraps the a2a-sdk client to send messages to purple agents and receive responses.
Maintains conversation context across turns within a single episode.
"""

from __future__ import annotations

import json
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart, DataPart

DEFAULT_TIMEOUT = 120  # seconds — generous for slow models


def _create_message(text: str, context_id: str | None = None) -> Message:
    return Message(
        kind="message",
        role=Role.user,
        parts=[Part(root=TextPart(kind="text", text=text))],
        message_id=uuid4().hex,
        context_id=context_id,
    )


def _merge_parts(parts: list[Part]) -> str:
    chunks = []
    for part in parts:
        if isinstance(part.root, TextPart):
            chunks.append(part.root.text)
        elif isinstance(part.root, DataPart):
            chunks.append(json.dumps(part.root.data, indent=2))
    return "\n".join(chunks)


async def _send_message(
    message: str,
    base_url: str,
    context_id: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    async with httpx.AsyncClient(timeout=timeout) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=base_url)
        agent_card = await resolver.get_agent_card()
        # streaming=True to receive intermediate status updates as they arrive.
        # With streaming=False only the final completed snapshot is returned,
        # which has task.status.message=None when the agent used update_status mid-flight.
        config = ClientConfig(httpx_client=httpx_client, streaming=True)
        client = ClientFactory(config).create(agent_card)

        outbound = _create_message(message, context_id=context_id)
        outputs: dict = {"response": "", "context_id": None}

        async for event in client.send_message(outbound):
            match event:
                case Message() as msg:
                    outputs["context_id"] = msg.context_id
                    outputs["response"] += _merge_parts(msg.parts)
                case (task, update):
                    outputs["context_id"] = task.context_id
                    outputs["status"] = task.status.state.value
                    # Accumulate text from intermediate status updates only
                    # (artifacts are collected separately via TaskArtifactUpdateEvent)
                    if update and hasattr(update, "status") and update.status.message:
                        outputs["response"] += _merge_parts(update.status.message.parts)
                    # Only collect artifacts from artifact-specific update events
                    if update and hasattr(update, "artifact") and update.artifact:
                        outputs["response"] += _merge_parts(update.artifact.parts)
                case _:
                    pass

    return outputs


class Messenger:
    """
    Stateful A2A messenger.

    Maintains one conversation context per agent URL so that multi-turn
    episodes (Turn 0 → Turn 1 → … → Turn N) arrive in a single conversation
    thread, giving the purple agent access to its own prior reasoning.

    Call reset() between episodes to start a fresh conversation.
    """

    def __init__(self) -> None:
        self._context_ids: dict[str, str | None] = {}

    async def talk_to_agent(
        self,
        message: str,
        url: str,
        new_conversation: bool = False,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> str:
        """
        Send a message to a purple agent and return its text response.

        Args:
            message:          Prompt to send.
            url:              Purple agent base URL.
            new_conversation: If True, ignore any existing context (start fresh).
            timeout:          HTTP timeout in seconds.

        Returns:
            The agent's response text.

        Raises:
            RuntimeError: If the agent returns a non-completed status.
        """
        context_id = None if new_conversation else self._context_ids.get(url)
        outputs = await _send_message(message, url, context_id=context_id, timeout=timeout)

        status = outputs.get("status", "completed")
        if status != "completed":
            raise RuntimeError(f"Purple agent at {url} returned status={status!r}: {outputs}")

        self._context_ids[url] = outputs.get("context_id")
        return outputs["response"]

    def reset(self) -> None:
        """Clear all conversation contexts (call between episodes)."""
        self._context_ids.clear()
