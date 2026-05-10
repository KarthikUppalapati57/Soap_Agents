# Agent-1: SOAP Note Generator

## Overview

Agent-1 is the core clinical reasoning module in this project. It takes raw patient input (symptoms, history, and context) and converts it into structured **SOAP notes**:

* **S** → Subjective (patient-reported symptoms)
* **O** → Objective (clinical observations)
* **A** → Assessment (diagnosis reasoning)
* **P** → Plan (treatment suggestions)

This enables consistent medical documentation and supports downstream evaluation by other agents.

---

## Key Responsibilities

* Transform unstructured patient input into structured SOAP format
* Maintain logical clinical reasoning
* Generate consistent and evaluation-ready outputs
* Act as the primary input generator for later agents

---

## Architecture

```
User Input → Prompt Engineering → LLM → SOAP Formatter → JSON Output
```

### Components

* Prompt templates for structured reasoning
* LLM (OpenAI API) for generating responses
* Post-processing layer for formatting into JSON

---

## Project Structure

```
Agent-1/
│── main.py              # Entry point
│── prompts/             # Prompt templates
│── V1                   # This folder have all the agent codes
│── utils/               # Helper functions
│── results/             # Generated outputs
```

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KarthikUppalapati57/Soap_Agents.git
cd Soap_Agents/Agent-1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set API Key

```bash
we used the Ollama models
Used llama mistrail model
```

---

## Execution

Run the agent:

```bash
python main.py
```

If you want to run Agent-1 individually, you can execute the `run.py` script from the root folder:

```bash
uv run python Agent_1/V1/run.py
```

---

## Example

### Input

```json
{
  "patient_input": "I have a fever, headache, and body pain for 2 days"
}
```

### Output

```json
{
  "Subjective": "Patient reports fever, headache, and body pain for 2 days.",
  "Objective": "No vitals provided.",
  "Assessment": "Likely viral infection.",
  "Plan": "Rest, hydration, and symptomatic treatment."
}
```

---

## Output Storage

Results are stored in:

```
/results/v1_results.json
```

---

## Design Decisions

* Uses LLM-based reasoning instead of rule-based logic
* Outputs structured JSON for easy evaluation
* Focuses on prompt engineering for flexibility
