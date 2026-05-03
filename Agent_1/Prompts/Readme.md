# Prompts Module – Agent-1

##  Overview

The `prompts/` folder contains all prompt templates used by Agent-1 to guide the LLM in generating structured SOAP notes.

This layer is basically the **control center for model behavior** — it defines how raw patient input gets transformed into clinically structured output.

---

##  Purpose

* Standardize how the LLM interprets patient input
* Enforce SOAP format consistency (Subjective, Objective, Assessment, Plan)
* Reduce hallucinations through guided instructions
* Enable easy iteration without changing core logic

---

##  Prompt Design Strategy

The prompts are engineered to simulate clinician-style reasoning:

* **Instruction-driven** → Clearly tells the model what to generate
* **Structured output enforcement** → Forces SOAP format
* **Context-aware** → Incorporates patient symptoms and history
* **Minimal ambiguity** → Reduces inconsistent outputs

---

##  Folder Structure

```id="rx7qhv"
prompts/
│── prompt.txt        # trail prompt template for SOAP generation
│── zero_shot.txt        # zero shot prompt template for SOAP generation
│── one_shot.txt        # One shot prompt template for SOAP generation
│── Few_shot.txt        # Few shot prompt template for SOAP generation
```

---

##  How It Works

1. User input is received (symptoms, complaints, etc.)
2. The prompt template is loaded from this folder
3. Input is injected into the template
4. Final prompt is sent to the LLM
5. LLM returns structured SOAP response

---

##  Integration

This module is used directly by:

* `main.py` → Loads and formats prompts
* LLM API layer → Sends prompt for generation
* Output formatter → Ensures structured JSON

---

##  Why Prompts Matter

Prompt quality directly impacts:

* Clinical accuracy
* Logical consistency
* Output structure
* Evaluation scores in downstream agents

In short: **better prompts = better agent performance**

---
