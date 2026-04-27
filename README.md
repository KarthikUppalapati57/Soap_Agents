# Soap Agents

Multi-stage pipeline for **SOAP note generation** (local LLM), **quality evaluation** (local LLM), and **claim verification** (Google Gemini via [Google ADK](https://google.github.io/adk-docs/)).

## Prerequisites

| Requirement | Used for |
|-------------|----------|
| **Python 3.13+** | Project `requires-python` |
| **[uv](https://docs.astral.sh/uv/)** | Install and run (`uv sync`, `uv run`) |
| **[Ollama](https://ollama.com/)** running | Agent 1 (`mistral`) and Agent 2 (`llama3.2`) |
| **Google Gemini API key** | Agent 3 (ADK) — `GOOGLE_API_KEY` or `GEMINI_API_KEY` |

After installing Ollama, pull the models this repo calls:

```bash
ollama pull mistral
ollama pull llama3.2
```

## Install

From the repository root (`Soap_Agents/`):

```bash
uv sync
```

That creates/updates `.venv` and installs dependencies from `uv.lock`.

## Configuration

### API key (Agent 3)

Create a **`.env`** file at the **repository root** (and/or under `Agent_3/.env`). Either variable is accepted; the app normalizes to `GOOGLE_API_KEY` when needed:

```bash
# .env (example)
GOOGLE_API_KEY=your-key-here
# or: GEMINI_API_KEY=...
```

Optional: `Agent_3` model override — `AGENT3_GEMINI_MODEL` (default `gemini-2.5-flash`).

### Data and prompts

The main pipeline expects:

| Path | Purpose |
|------|---------|
| `data/clean_medsynth_final.json` | Input dataset (list of objects with at least `transcript`, `ground_truth`) |
| `prompts/A1_prompt.txt` | Template for SOAP generation (`{transcript}` placeholder) |
| `prompts/A2_prompt.txt` | Template for evaluation (`{transcript}`, `{generated}`) |

Sample prompt files are already under `prompts/`. Place or symlink your MedSynth (or compatible) JSON at `data/clean_medsynth_final.json`.

## Run the full pipeline (entry point)

Run from the **repository root** so imports resolve:

```bash
mkdir -p Output
uv run python main.py
```

`main.py` builds a `Pipeline` (`pipeline/pipeline.py`), streams results from `data/clean_medsynth_final.json`, and writes JSON to **`Output/claim_verification.json`**.

**Services that must be running:** Ollama (for Agents 1–2) and a valid Gemini key in `.env` (for Agent 3).

## Run pieces individually

### Agent interface only (small JSON loop)

Still from repo root:

```bash
uv run python -m pipeline.agent_interface
```

Uses `data/clean_medsynth_final.json` and stops after a few rows (see `if __name__ == "__main__"` in `pipeline/agent_interface.py`).

### Agent 3 only (batch on Agent 2–style JSON)

```bash
uv run python -m Agent_3.agent3 --limit 2
```

Override input/output:

```bash
uv run python -m Agent_3.agent3 \
  --input path/to/v2_results.json \
  --output Agent_3/results/v2_with_claims.json
```

### ADK web UI (Agent 3)

```bash
cd Agent_3
uv run adk web agents
```

## Project layout (high level)

- **`main.py`** — Orchestrates `Pipeline` over `data/`.
- **`pipeline/`** — `Pipeline` class and `AgentInterface` (Agents 1 → 2 → 3).
- **`Agent_1/`** — SOAP generation (`V1/generate.py`, Ollama).
- **`Agent_2/v2/OpenAI_health_Benchmark/`** — SOAP evaluation (`agent2.py`, Ollama).
- **`Agent_3/`** — Claim verification (Google ADK); see `Agent_3/README.md` for agent-specific notes.

## Troubleshooting

- **`ModuleNotFoundError` for `Agent_3` / `pipeline`:** Run commands from the repo root, or use `uv run python -m …` as shown above.
- **Ollama connection errors:** Ensure `ollama serve` (or the Ollama app) is running and the required models are pulled.
- **Agent 3 auth errors:** Confirm `.env` is at the repo root or `Agent_3/` and contains a valid key.
