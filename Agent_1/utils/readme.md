#  Utils Module – Agent-1

##  Overview

The `utils/` folder contains helper functions used across Agent-1.
Currently, it focuses on **dataset loading and preprocessing**, acting as the entry point for accessing structured medical data.

---

##  Purpose

* Centralize dataset loading logic
* Keep core pipeline (`v1/`) clean and modular
* Provide reusable utilities for future extensions
* Ensure consistent data format across the project

---

##  Files

### `loader.py`

Handles loading of the cleaned MedSynth dataset.

---

##  Functionality

### `load_clean_medsynth()`

```python
load_clean_medsynth(path="data/clean_medsynth_final.json")
```

 What it does:

* Reads the dataset from a JSON file
* Parses it into Python objects
* Returns structured data for processing

 Returns:

* A list of dictionaries, each containing:

  * `transcript` → doctor–patient conversation
  * `ground_truth` → reference SOAP note

 Implementation:

* Uses standard JSON loading
* No transformation applied (clean dataset assumed) 

---

##  Data Format

Each dataset entry looks like:

```json id="x1g5dp"
{
  "transcript": "[Doctor]: ... [Patient]: ...",
  "ground_truth": "Subjective: ... Objective: ... Assessment: ... Plan: ..."
}
```

---

##  How It Fits in the Pipeline

```id="g38n0w"
Dataset (JSON) → utils.loader → v1/run.py → generate.py → results/
```

* `utils/` provides the data
* `v1/` consumes it for generation
* `results/` stores outputs

---

##  Design Decisions

### Minimal Abstraction

* Keeps logic simple and readable
* Avoids unnecessary preprocessing layers

---

### Reusability

* Can be reused across:

  * Future agent versions (v2, v3)
  * Evaluation pipelines
  * Data analysis scripts

---

##  Limitations

* No validation of dataset structure
* No error handling for corrupted/missing files
* No support for multiple dataset formats
* Hardcoded default path

---
