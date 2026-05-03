#  v1 – SOAP Generation Pipeline (Agent-1)

##  Overview

The `v1/` folder contains the **first version of the SOAP note generation pipeline** for Agent-1.

It defines how:

* Raw medical transcripts → get converted into → structured SOAP notes
* Prompts are applied
* Model inference is executed
* Results are stored for evaluation

---

##  Architecture (High-Level Flow)

```
Transcript → Prompt Template → LLM (Mistral via Ollama) → SOAP Output → JSON Results
```

---

##  Files in This Folder

### 1. `generate.py`

Core generation logic for SOAP notes.

 Responsibilities:

* Load prompt template
* Inject transcript into prompt
* Apply optional prompt optimizations
* Call LLM (via Ollama)
* Return generated SOAP note

 Key function:

```python
generate_soap_v1(transcript, prompt_path="v1/prompt.txt", prompt_optimizations=None)
```

 Uses:

* Model: `mistral` (can switch to `llama3`)
* Temperature: `0.0` (deterministic output)

 Insight:

* Prompt is dynamically constructed using placeholders:

  * `{transcript}`
  * `{prompt_optimizations}` 

---

### 2. `run.py`

Execution script to generate SOAP notes for multiple samples.

 Responsibilities:

* Load dataset
* Loop through samples
* Call `generate_soap_v1()`
* Collect outputs
* Save results to JSON

 Key function:

```python
run_v1(num_samples=50)
```

 Output:

* Saves results as:

```
v1_results.json
```

 Data source:

```
data/clean_medsynth_final.json
```

 Output structure:

```json
{
  "id": 0,
  "transcript": "...",
  "generated": "...",
  "ground_truth": "..."
}
```



---

### 3. `prompt.txt` (structure) [We have all the prompts inside the Promots folder]

* Prompt template used for SOAP generation
* Includes structured instructions for:

  * Subjective
  * Objective
  * Assessment
  * Plan

 Note:

* This file is required for execution
* Uses placeholders:

  * `{transcript}`
  * `{prompt_optimizations}`

---

##  Execution Workflow

### Step 1: Load Dataset

* Reads medical transcripts from:

```
data/clean_medsynth_final.json
```

---

### Step 2: Generate SOAP Notes

For each sample:

* Extract transcript
* Inject into prompt
* Call LLM via Ollama
* Generate structured SOAP note

---

### Step 3: Store Results

* Save all outputs into:

```
v1_results.json
```

---

##  How to Run

### Run full pipeline:

```bash
python v1/run.py
```

### Customize number of samples:

```python
run_v1(num_samples=100)
```

---

##  Configuration

| Parameter   | Value                     | Purpose              |
| ----------- | ------------------------- | -------------------- |
| Model       | mistral                   | LLM for generation   |
| Temperature | 0.0                       | Deterministic output |
| Dataset     | clean_medsynth_final.json | Input transcripts    |
| Output      | v1_results.json           | Generated results    |

---

##  Design Decisions

### Deterministic Outputs

* Temperature = 0 ensures reproducibility

---

### Prompt-Driven Generation

* No fine-tuning
* Entire behavior controlled via prompt engineering

---

### Modular Design

* `generate.py` → handles inference
* `run.py` → handles execution & batching

---

##  Limitations (v1)

* No evaluation inside pipeline (handled separately)
* No error recovery beyond simple try/except
* No batch optimization (sequential processing)
* No metadata tracking (model version, prompt version)
* Prompt sensitivity can affect output quality

---

##  Key Insight

`v1` is your **baseline generation engine**.

It establishes:

> Prompt → Model → Output → Stored Results

Everything downstream (evaluation, optimization, scaling) builds on top of this layer.

---
