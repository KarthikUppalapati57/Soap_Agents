# Soap Agents

Multi-stage pipeline that turns doctor-patient transcripts into SOAP notes and then validates them. Each transcript flows through:

1. **Agent 1** — Generate a SOAP note from the transcript with a local Ollama model (`mistral`).
2. **Agent 2** — Score the SOAP against the transcript with another local Ollama model (`llama3.2`) and parse the result into a structured benchmark.
3. **PrimeKG context** — Extract clinical terms from the SOAP with GLiNER and pull related triples from the PrimeKG knowledge graph (`Agent_3/MKG`).
4. **Agent 3** — Verify each clinical claim against the transcript and the PrimeKG context with Google Gemini via [Google ADK](https://google.github.io/adk-docs/).
5. **Prompt optimizer** — If Agent 3 finds unsupported claims, the pipeline rewrites the Agent 1 prompt and retries (up to 4 iterations per item).
6. **GT Validator** (`Agent_GT_Validator`) — After the pipeline, compare the final generated SOAP against the dataset's ground truth: clinical entity recall (symptoms / meds / labs), structural alignment (token & ROUGE-L F1), an LLM expert judge (Gemini), and per-addition transcript evidence checks.
7. **Analysis & viewer** — Export CSVs, generate charts, and browse all results in a static HTML viewer.

## Prerequisites

| Requirement | Used for |
|-------------|----------|
| **Python 3.13+** | `pyproject.toml` `requires-python` |
| **[uv](https://docs.astral.sh/uv/)** | Install and run (`uv sync`, `uv run`) |
| **[Ollama](https://ollama.com/)** running | Agent 1 (`mistral`) and Agent 2 (`llama3.2`) |
| **Google Gemini API key** | Agent 3 (ADK) and GT Validator judge — `GOOGLE_API_KEY` or `GEMINI_API_KEY` |
| **PrimeKG CSV** *(optional)* | PrimeKG context for Agent 3 — at `Agent_3/mkg/kg.csv` or `PRIMEKG_CSV` |

After installing Ollama, pull the models the pipeline calls:

```bash
ollama pull mistral
ollama pull llama3.2
```

## Install

From `Soap_Agents/`:

```bash
uv sync
```

That creates/updates `.venv` and installs dependencies from `uv.lock`.

## Configuration

### API key

Create a `.env` at the repo root (and/or `Agent_3/.env`). Either variable is accepted; the runners normalize to `GOOGLE_API_KEY`:

```bash
GOOGLE_API_KEY=your-key-here
# or: GEMINI_API_KEY=...
```

### Useful environment variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `AGENT3_GEMINI_MODEL` | Agent 3 Gemini model | `gemini-2.5-flash` |
| `GTJUDGE_GEMINI_MODEL` | GT Validator judge model | `gemini-2.5-flash` |
| `MAX_TOKENS` | Cap on Gemini output tokens | `2048` |
| `PRIMEKG_CSV` | Path to PrimeKG triples CSV | `Agent_3/mkg/kg.csv` |
| `MKG_KG_HOPS` | PrimeKG expansion hops | `1` |
| `MKG_KG_MAX_ROWS` | PrimeKG row pool per query | `2000` |
| `MKG_GLINER_MAX_TERMS` | Max GLiNER terms used | `12` |
| `MKG_PRIMEKG_QUERY_TERMS` | Terms queried per item | `8` |
| `MKG_GLINER_THRESHOLD` | GLiNER score threshold | `0.35` |

### Data and prompts

Run `preprocessing.ipynb` first. It loads the upstream MedSynth-style dataset, applies cleaning, and writes the pipeline inputs:

- `data/clean_medsynth_final.json` (used by `main.py`)
- `data/clean_medsynth_final.csv` (convenience export)

The prompt templates live under `prompts/` (`A1_prompt.txt`, `A1_few_shot.txt`, `A1_one_shot.txt`, `A1_zero_shot.txt`, `A2_prompt.txt`, `A3_optimizer_prompt.txt`). `A1_prompt.txt` is the default Agent 1 template; `A3_optimizer_prompt.txt` drives the prompt optimizer.

## Run the main pipeline

From `Soap_Agents/`:

```bash
uv run python main.py --limit 50
```

`main.py` builds a `Pipeline` (`pipeline/pipeline.py`), iterates over `data/clean_medsynth_final.json`, and for every item runs Agent 1 → Agent 2 → PrimeKG context → Agent 3, retrying with an optimized prompt while Agent 3 reports unsupported claims (up to 4 iterations). For each item it writes a folder `Output/item_####/` containing:

| File | Contents |
|------|----------|
| `meta.json` | `index`, `iterations` (how many Agent 1 retries ran) |
| `source.json` | Original `transcript` and `ground_truth` |
| `agent3_result.json` | Final generated SOAP, Agent 2 benchmark scores, Agent 3 claim verification |
| `prompt_optimizations.json` | Rationales appended by the prompt optimizer per iteration |

Services that must be running: Ollama (Agents 1–2) and a valid Gemini key in `.env` (Agent 3 + optimizer).

## Validate against ground truth

After the main pipeline produces `Output/item_####/`, run:

```bash
uv run python -m Agent_GT_Validator.main
```

Useful flags:

```bash
uv run python -m Agent_GT_Validator.main \
  --output-dir Output \
  --analysis-dir Output/analysis \
  --limit 10 \
  --judge-model gemini-2.5-flash \
  --skip-llm   # runs offline metrics only (no Gemini judge)
```

This adds `Output/item_####/gt_validator.json` per item and writes `Output/analysis/gt_validator_summary.csv`. Each `gt_validator.json` includes:

- Extracted entities for ground truth and generated SOAP (all / symptoms / meds / labs)
- Clinical entity recall numbers (overlap / GT-only / Gen-only)
- Structural alignment per SOAP section (token F1, ROUGE-L F1)
- Expert judge output: overall grade, summary, omissions, additions, hallucinations / unjustified inferences, per-section grades
- `addition_evidence`: per-addition `supported` / `unsupported` / `unknown` flag against the transcript

## Analysis and visualization

All scripts run from `Soap_Agents/`:

```bash
# Flat CSV with per-item metrics across the whole pipeline.
uv run python scripts/export_output_metrics_csv.py

# Charts for Agent 1–3 outputs (Output/analysis/).
uv run python scripts/visualize_output_analytics.py

# Charts for GT validator outputs (Output/analysis/gt_validator/).
uv run python scripts/visualize_gt_validator_analytics.py

# Distribution of unsupported Agent 3 claims by SOAP section.
uv run python scripts/plot_agent3_unsupported_by_section.py
```

## Browse results in the viewer

`Output/viewer.html` is a static page that loads the per-item JSON over HTTP. Serve `Output/` and open the page:

```bash
cd Output
python3 -m http.server 8000
# open http://localhost:8000/viewer.html
```

Per item it shows: Overview, SOAP Compare, Transcript, Ground Truth SOAP, Generated SOAP, Entities, Judge, Claims, Raw JSON. The left sidebar also has two **global views** — `All Hallucinations` and `All Omissions` — that aggregate the GT Validator overall lists across every loaded item, with direct links to the source `gt_validator.json` files.

## Run pieces individually

### Agent 3 only (batch on Agent 2-style JSON)

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

- **`preprocessing.ipynb`** — Build `data/clean_medsynth_final.{json,csv}` from the upstream dataset.
- **`main.py`** — CLI entry point; runs `Pipeline` over `data/`.
- **`pipeline/`** — `Pipeline` (iteration + per-item output) and `AgentInterface` (Agents 1 → 2 → PrimeKG → 3 + prompt optimizer).
- **`Agent_1/`** — SOAP generation via Ollama (`Agent_1/V1/`).
- **`Agent_2/`** — SOAP scoring via Ollama (`Agent_2/v2/OpenAI_health_Benchmark/`).
- **`Agent_3/`** — Claim verification via Gemini/ADK; `Agent_3/MKG/` holds GLiNER term extraction and the PrimeKG explorer.
- **`Agent_GT_Validator/`** — Post-hoc comparison of generated SOAP to ground truth (entities, structure, judge, evidence).
- **`scripts/`** — CSV exports and plotting (`plot_style.py` is the shared matplotlib style).
- **`prompts/`** — Prompt templates for Agent 1 / Agent 2 / optimizer.
- **`data/`** — Cleaned MedSynth-style inputs.
- **`Output/`** — Per-item pipeline outputs (`item_####/`), aggregated analysis (`analysis/`), and `viewer.html`.

## Troubleshooting

- **`ModuleNotFoundError` for `Agent_3` / `pipeline` / `Agent_GT_Validator`** — Run from `Soap_Agents/` or use `uv run python -m ...` as shown above.
- **Ollama connection errors** — Ensure `ollama serve` (or the Ollama app) is running and `mistral` + `llama3.2` are pulled.
- **Agent 3 / GT Validator auth errors** — Confirm `.env` is at the repo root or `Agent_3/` and contains a valid Gemini key.
- **PrimeKG context unavailable** — The pipeline still runs without it; either set `PRIMEKG_CSV` or place the CSV at `Agent_3/mkg/kg.csv`.
- **GT Validator without an API key** — Use `--skip-llm` to compute the offline metrics only.
