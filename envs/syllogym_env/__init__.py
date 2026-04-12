"""
SylloGym — Multi-turn Legal Reasoning Environment.

JudgeEnv: the agent plays a judge who receives case facts turn by turn
and must revise their ruling as the case evolves. Twelve domains of US
law, procedurally generated episodes, deterministic Python verifiers.

Example (standalone):
    >>> from syllogym_env import SylloGymEnv
    >>> env = SylloGymEnv()
    >>> obs = env.reset()
    >>> result = env.review_document("arrest_report")
    >>> result = env.conclude("Yes")
    >>> print(env.reward)

Example (MCP via OpenEnv client):
    >>> from openenv.core.mcp_client import MCPToolClient
    >>> client = MCPToolClient("http://localhost:8000")
    >>> tools = client.list_tools()
    >>> result = client.call_tool("review_document", name="arrest_report")
"""

from .server.core.investigation_env import SylloGymEnv
from .server.core.judge_environment import JudgeAction, JudgeObservation
from .judge_env import JudgeEnv, JudgeObs
from .models import SylloState

__all__ = ["SylloGymEnv", "JudgeEnv", "JudgeObs", "JudgeAction", "JudgeObservation", "SylloState"]
