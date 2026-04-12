"""
server.py
---------
SylloGym Green Agent — A2A server entry point.

Exposes a single skill: evaluate an A2A-compatible purple agent on SylloGym
multi-turn legal reasoning episodes.

Usage:
    python src/server.py --host 0.0.0.0 --port 9009 --card-url http://your-ip:9009/
"""

import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor


def main() -> None:
    parser = argparse.ArgumentParser(description="SylloGym Green Agent A2A server")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str, help="Public URL to advertise in the agent card")
    args = parser.parse_args()

    skill = AgentSkill(
        id="syllogym_eval",
        name="SylloGym Legal Reasoning Evaluation",
        description=(
            "Evaluates a purple agent on SylloGym — a multi-turn legal reasoning environment. "
            "The agent plays a judge who receives case facts turn by turn, including plot twists "
            "that may reverse the previous conclusion. Covers 9 domains of US law (diversity "
            "jurisdiction, UCC, Miranda, consideration, mens rea, Terry stop, and more). "
            "Returns per-task accuracy and mean reward across all evaluated episodes."
        ),
        tags=["legal-reasoning", "multi-turn", "rl-environment", "openenv", "grpo"],
        examples=[],
    )

    agent_card = AgentCard(
        name="SylloGym Green Agent",
        description=(
            "Green agent wrapper for the SylloGym multi-turn legal reasoning environment. "
            "Orchestrates evaluation battles: resets episodes, sends observations to the purple "
            "agent, submits its answers to the environment, and reports dense rewards per turn."
        ),
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill],
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
