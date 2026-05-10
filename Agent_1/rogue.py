import json
import os
from rouge_score import rouge_scorer

INPUT_FILES = {
    "zero_shot": r"C:\Users\ukart\OneDrive - University of Tennessee\M\4th SEm\Project\codes\codes\results\v1_rouge_Zero-Shot_Final_results.json",
    "one_shot":  r"C:\Users\ukart\OneDrive - University of Tennessee\M\4th SEm\Project\codes\codes\results\v1_rouge_One-Shot_Final_results.json",
    "few_shot":  r"C:\Users\ukart\OneDrive - University of Tennessee\M\4th SEm\Project\codes\codes\results\v1_rouge_Few-Shot_Final_results.json"
}

OUT_DIR = "results"


scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rouge2', 'rougeL'],
    use_stemmer=True
)


def evaluate_raw_dataset(data, label):

    results = []

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {label.upper()}")
    print(f"{'=' * 60}")

    for sample in data:

        sid = sample.get("id", "N/A")
        generated = sample.get("generated", "")
        ground_truth = sample.get("ground_truth", "")

        scores = scorer.score(ground_truth, generated)

        r1 = round(scores["rouge1"].fmeasure, 4)
        r2 = round(scores["rouge2"].fmeasure, 4)
        rl = round(scores["rougeL"].fmeasure, 4)

        results.append({
            "id": sid,
            "rouge1": r1,
            "rouge2": r2,
            "rougeL": rl
        })

        print(f"Sample {sid} → R1: {r1} | R2: {r2} | RL: {rl}")

    avg_r1 = sum(r["rouge1"] for r in results) / len(results)
    avg_r2 = sum(r["rouge2"] for r in results) / len(results)
    avg_rl = sum(r["rougeL"] for r in results) / len(results)

    summary = {
        "prompt_type": label,
        "average": {
            "rouge1": round(avg_r1, 4),
            "rouge2": round(avg_r2, 4),
            "rougeL": round(avg_rl, 4)
        },
        "samples": results
    }

    return summary

def read_existing_rouge(data, label):

    print(f"\n{'=' * 60}")
    print(f"Reading Existing ROUGE Scores: {label.upper()}")
    print(f"{'=' * 60}")

    avg = data.get("average", {})

    print(f"ROUGE-1: {avg.get('rouge1', 0)}")
    print(f"ROUGE-2: {avg.get('rouge2', 0)}")
    print(f"ROUGE-L: {avg.get('rougeL', 0)}")

    return {
        "prompt_type": label,
        "average": avg,
        "samples": data.get("samples", [])
    }


def run_all():

    os.makedirs(OUT_DIR, exist_ok=True)

    all_summaries = []

    for label, file_path in INPUT_FILES.items():

        if not os.path.exists(file_path):
            print(f"\nFile not found: {file_path}")
            continue

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)


        if isinstance(data, list):

            # RAW DATASET
            summary = evaluate_raw_dataset(data, label)

        elif isinstance(data, dict) and "average" in data:

            # ALREADY EVALUATED FILE
            summary = read_existing_rouge(data, label)

        else:

            print(f"\nUnsupported JSON structure in: {file_path}")
            continue

        out_path = os.path.join(
            OUT_DIR,
            f"rouge_{label}.json"
        )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print(f"\nSaved: {out_path}")

        all_summaries.append(summary)

    combined = {}

    for s in all_summaries:

        combined[s["prompt_type"]] = s["average"]

    combined_path = os.path.join(
        OUT_DIR,
        "Final_rouge_summary.json"
    )

    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)

    print(f"\nFINAL SUMMARY SAVED → {combined_path}")

    print("\nCOMPARISON TABLE")
    print("-" * 60)

    for k, v in combined.items():

        print(
            f"{k:10} → "
            f"R1: {v['rouge1']} | "
            f"R2: {v['rouge2']} | "
            f"RL: {v['rougeL']}"
        )

if __name__ == "__main__":
    run_all()