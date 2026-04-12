# SylloGym Green Agent

A2A-compatible green agent wrapper for [SylloGym](https://huggingface.co/spaces/farffadet/syllogym-env) — a multi-turn legal reasoning environment.

Evaluates any A2A-compatible purple agent on all 35 SylloGym tasks across 9 domains of US law.

## Quick start

```bash
# Install
cd green_agent
uv sync

# Run
uv run python src/server.py --host 0.0.0.0 --port 9009
```

## Request format

Send a JSON message to the agent with this structure:

```json
{
    "participants": {
        "solver": "http://your-purple-agent:9019/"
    },
    "config": {
        "episodes_per_task": 5,
        "task_mode": "mixed",
        "env_url": "https://farffadet-syllogym-env.hf.space"
    }
}
```

### Config options

| Key | Default | Description |
|-----|---------|-------------|
| `episodes_per_task` | `5` | Episodes per task |
| `task_mode` | `"mixed"` | `"mixed"` (all 35 tasks) or `"single"` |
| `task_name` | — | Required if `task_mode="single"` (e.g. `"diversity_3"`) |
| `env_url` | HF Space URL | SylloGym environment URL |

## Docker

```bash
docker build -t syllogym-green-agent .
docker run -p 9009:9009 syllogym-green-agent
```

## Output

The agent returns a structured artifact with:
- `overall_accuracy` — fraction of fully-correct episodes
- `overall_mean_reward` — mean `turns_correct / total_turns` across all episodes
- `per_task` — per-task breakdown with accuracy and mean reward
