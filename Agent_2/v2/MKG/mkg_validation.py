import json
import os
import sys
import requests
from pathlib import Path
from dotenv import load_dotenv

# Allow `python mkg_validation.py` from this folder to resolve `Agent_3.MKG`.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Agent_3.MKG import extract_medical_terms

# ── Load API Key ───────────────────────────────────
load_dotenv("../../.env")
API_KEY = os.getenv("UMLS_API_KEY")

# ── Paths ──────────────────────────────────────────
V2_PATH    = "../OpenAI health Benchmark/results/v2_results.json"
OUT_PATH   = "../OpenAI health Benchmark/results/mkg_results.json"
CHECKPOINT = "../OpenAI health Benchmark/results/mkg_checkpoint.json"


def extract_terms(text: str) -> list[str]:
    """Medical surface forms via GLiNER (same helper as PrimeKG / agent_interface)."""
    threshold = float(os.getenv("MKG_GLINER_THRESHOLD", "0.35"))
    terms = extract_medical_terms(text, threshold=threshold)
    max_n = (os.getenv("MKG_UMLS_MAX_TERMS") or "").strip()
    if max_n.isdigit():
        n = int(max_n)
        if n > 0:
            terms = terms[:n]
    return terms


# ── Check Term Against UMLS API ────────────────────
def check_umls(term):
    url    = "https://uts-ws.nlm.nih.gov/rest/search/current"
    params = {
        "string":       term,
        "apiKey":       API_KEY,
        "returnIdType": "concept"
    }
    try:
        r       = requests.get(url, params=params, timeout=10)
        data    = r.json()
        results = data.get("result", {}).get("results", [])
        if results and results[0].get("ui") != "NONE":
            return True
        return False
    except Exception:
        return False

# ── Validate All Terms ─────────────────────────────
def validate_terms(terms):
    valid   = []
    invalid = []
    for term in terms:
        if check_umls(term):
            valid.append(term)
        else:
            invalid.append(term)
    return valid, invalid

# ── Main ───────────────────────────────────────────
def run():
    with open(V2_PATH, "r") as f:
        v2_data = json.load(f)

    # Resume from checkpoint
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming — {len(done_ids)} samples done.")
    else:
        results  = []
        done_ids = set()

    for sample in v2_data:
        i         = sample["id"]
        generated = sample["generated"]

        if i in done_ids:
            continue

        print(f"Processing sample {i:02d}...")

        # Step 1 — Extract terms
        terms = extract_terms(generated)
        print(f"  Extracted {len(terms)} terms")

        # Step 2 — Validate against UMLS
        valid, invalid = validate_terms(terms)

        # Step 3 — Calculate score
        total     = len(terms)
        mkg_score = round(len(valid) / total, 4) if total > 0 else 0.0

        result_entry = {
            "id":            i,
            "total_terms":   total,
            "valid_terms":   valid,
            "invalid_terms": invalid,
            "valid_count":   len(valid),
            "invalid_count": len(invalid),
            "mkg_score":     mkg_score
        }

        results.append(result_entry)

        print(f"  Sample {i:02d} → "
              f"Total: {total} | "
              f"Valid: {len(valid)} | "
              f"Invalid: {len(invalid)} | "
              f"Score: {mkg_score}")

        # Save checkpoint
        with open(CHECKPOINT, "w") as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    # ── Summary ───────────────────────────────────
    avg_score = round(
        sum(r["mkg_score"] for r in results) / len(results), 4
    ) if results else 0.0

    avg_valid = round(
        sum(r["valid_count"] for r in results) / len(results), 2
    )

    avg_invalid = round(
        sum(r["invalid_count"] for r in results) / len(results), 2
    )

    print(f"\n{'─'*50}")
    print(f"MKG VALIDATION COMPLETE")
    print(f"{'─'*50}")
    print(f"Total Samples      : {len(results)}")
    print(f"Avg Valid Terms    : {avg_valid}")
    print(f"Avg Invalid Terms  : {avg_invalid}")
    print(f"Avg MKG Score      : {avg_score}")
    print(f"{'─'*50}")
    print(f"Results saved to   : {OUT_PATH}")

if __name__ == "__main__":
    run()
