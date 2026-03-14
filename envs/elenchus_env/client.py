"""
ElenchusEnv Client.

Typed client for connecting to a running ElenchusEnv server.

ElenchusEnv uses MCP tools for multi-turn agentic episodes. The agent calls
tools (check_rule, get_facts, derive, submit_answer) via CallToolAction.

Example:
    >>> from elenchus_env import ElenchusEnv
    >>> from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction
    >>>
    >>> env = ElenchusEnv(base_url="http://localhost:8000")
    >>> env.connect()
    >>>
    >>> # Reset: receive the problem statement
    >>> result = env.reset()
    >>> print(result.observation)
    >>>
    >>> # Discover tools
    >>> result = env.step(ListToolsAction())
    >>> print([t.name for t in result.observation.tools])
    >>>
    >>> # Multi-turn: explore the problem
    >>> result = env.step(CallToolAction(tool_name="check_rule", arguments={}))
    >>> print(result.observation.result)
    >>>
    >>> result = env.step(CallToolAction(tool_name="get_facts", arguments={}))
    >>> print(result.observation.result)
    >>>
    >>> result = env.step(CallToolAction(
    ...     tool_name="derive",
    ...     arguments={"statement": "Since all A's are B's and entity is an A..."}
    ... ))
    >>>
    >>> # Submit final answer
    >>> result = env.step(CallToolAction(
    ...     tool_name="submit_answer",
    ...     arguments={"answer": "Yes"}
    ... ))
    >>> print("Reward:", result.reward)  # 1.0 or 0.0
    >>> env.disconnect()
"""

from __future__ import annotations

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from .models import ElenchusObservation, ElenchusState


class ElenchusEnv(EnvClient[CallToolAction, ElenchusObservation, ElenchusState]):
    """
    Client for the ElenchusEnv multi-turn agentic reasoning environment.

    Uses MCP tools: check_rule, get_facts, derive, submit_answer.
    Each episode ends when submit_answer is called or max_steps is exceeded.

    Args:
        base_url: URL of the running ElenchusEnv server.

    Example:
        >>> env = ElenchusEnv("http://localhost:8000")
        >>> env.connect()
        >>> result = env.reset()
        >>> # ... tool calls ...
        >>> result = env.step(CallToolAction(tool_name="submit_answer", arguments={"answer": "Yes"}))
        >>> print(result.reward)
        >>> env.disconnect()
    """

    def _step_payload(self, action: CallToolAction) -> dict:
        return {
            "type": "call_tool",
            "tool_name": action.tool_name,
            "arguments": action.arguments,
        }

    def _parse_result(self, payload: dict) -> StepResult[ElenchusObservation]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = bool(payload.get("done", False))

        # After a tool step the server returns a raw CallToolObservation
        # (tool_name + result). Wrap it back into our ElenchusObservation.
        if "result" in obs_data and "problem" not in obs_data:
            mcp_result = obs_data.get("result")
            tool_text = None
            if mcp_result and isinstance(mcp_result.get("content"), list):
                contents = mcp_result["content"]
                if contents:
                    tool_text = contents[0].get("text")

            # MCPEnvironment doesn't propagate reward/done for tool steps.
            # Infer them from submit_answer tool result text.
            tool_name = obs_data.get("tool_name")
            if tool_name == "submit_answer" and tool_text:
                done = True
                reward = 1.0 if "Correct!" in tool_text else 0.0

            obs = ElenchusObservation(
                tool_name=tool_name,
                tool_result=tool_text,
                reward=reward,
                done=done,
            )
        else:
            obs_kwargs: dict = {
                "problem": obs_data.get("problem", ""),
                "valid_answers": obs_data.get("valid_answers", []),
                "steps_used": obs_data.get("steps_used", 0),
                "max_steps": obs_data.get("max_steps", 8),
                "derived_facts": obs_data.get("derived_facts", []),
                "reward": reward,
                "done": done,
            }
            if obs_data.get("task_name") is not None:
                obs_kwargs["task_name"] = obs_data["task_name"]
            if obs_data.get("difficulty") is not None:
                obs_kwargs["difficulty"] = obs_data["difficulty"]
            obs = ElenchusObservation(**obs_kwargs)

        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: dict) -> ElenchusState:
        return ElenchusState(
            task_name=payload.get("task_name", ""),
            task_mode=payload.get("task_mode", "mixed"),
            total_correct=payload.get("total_correct", 0),
            total_steps=payload.get("total_steps", 0),
            total_episodes=payload.get("total_episodes", 0),
        )
