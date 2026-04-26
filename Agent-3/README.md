# Agent 3 — transcript-grounded SOAP claim verification (Google ADK)

Uses [uv](https://docs.astral.sh/uv/) for the environment.

```bash
cd Agent-3
uv sync
cp .env.example .env   # then set GOOGLE_API_KEY or GEMINI_API_KEY / gemini_api_key
```

Process Agent-2 `v2_results.json` sequentially (one row at a time):

```bash
uv run python agent3.py --limit 2
uv run python agent3.py --input ../Agent-2/v2/OpenAI health\ Benchmark/v2_results.json --output results/v2_with_claims.json
```

ADK Web UI (each subdirectory under `agents/` is one agent app):

```bash
uv run adk web agents
```
