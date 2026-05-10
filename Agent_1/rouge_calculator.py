import json
import os
from rouge_score import rouge_scorer
 
V1_PATH  = "C:\\Users\\ukart\\OneDrive - University of Tennessee\\M\\4th SEm\\Project\\codes\\codes\\v1_Final_results_Few_shot.json"
OUT_DIR  = "results"
OUT_PATH = "results/v1_rouge_Few-Shot_Final_results.json"
 
def run():
    os.makedirs(OUT_DIR, exist_ok=True)
 
    with open(V1_PATH, "r") as f:
        v1_data = json.load(f)
 
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    results = []
 
    for sample in v1_data:
        i            = sample["id"]
        generated    = sample["generated"]
        Reference_Text = sample["reference_text"]
 
        scores = scorer.score(Reference_Text, generated)
 
        results.append({
            "id":       i,
            "rouge1":   round(scores["rouge1"].fmeasure, 4),
            "rouge2":   round(scores["rouge2"].fmeasure, 4),
            "rougeL":   round(scores["rougeL"].fmeasure, 4),
        })
 
        print(f"Sample {i:02d} → ROUGE-1: {scores['rouge1'].fmeasure:.4f} | ROUGE-2: {scores['rouge2'].fmeasure:.4f} | ROUGE-L: {scores['rougeL'].fmeasure:.4f}")
 
    # Overall averages
    avg_r1 = sum(r["rouge1"] for r in results) / len(results)
    avg_r2 = sum(r["rouge2"] for r in results) / len(results)
    avg_rl = sum(r["rougeL"] for r in results) / len(results)
 
    output = {
        "average": {
            "rouge1": round(avg_r1, 4),
            "rouge2": round(avg_r2, 4),
            "rougeL": round(avg_rl, 4)
        },
        "samples": results
    }
 
    with open(OUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
 
    print(f"\n{'─'*50}")
    print(f"Done. Results saved to: {OUT_PATH}")
    print(f"  Average ROUGE-1: {avg_r1:.4f}")
    print(f"  Average ROUGE-2: {avg_r2:.4f}")
    print(f"  Average ROUGE-L: {avg_rl:.4f}")
 
if __name__ == "__main__":
    run()