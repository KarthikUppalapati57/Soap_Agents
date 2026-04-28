import json
import os
import ollama

# ── Paths ──────────────────────────────────────────
V1_PATH = "v2/v1_results.json"
PROMPT     = "v2/prompt.txt"
OUT_DIR    = "results"
OUT_PATH   = "results/v2_results.json"
CHECKPOINT = "results/checkpoint.json"

# ── Load Prompt ────────────────────────────────────
def load_prompt():
    with open(PROMPT, "r") as f:
        return f.read()

# ── Call LLM ───────────────────────────────────────
def evaluate_soap(prompt_template, transcript, generated):
    prompt = (
        prompt_template
        .replace("{transcript}", transcript)
        .replace("{generated}", generated)
    )
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0
        }
    )
    return response["message"]["content"].strip()

# ── Parse LLM Output ───────────────────────────────
def parse_output(raw):
    scores = {
        "accuracy_total_claims":           None,
        "accuracy_correct":                None,
        "accuracy_incorrect":              None,
        "accuracy_score":                  None,
        "accuracy_reason":                 "",

        "completeness_total_details":      None,
        "completeness_captured":           None,
        "completeness_missing":            None,
        "completeness_score":              None,
        "completeness_reason":             "",

        "communication_total_issues":      None,
        "communication_quality_score":     None,
        "communication_quality_reason":    "",

        "context_total_details":           None,
        "context_captured":                None,
        "context_missing":                 None,
        "context_awareness_score":         None,
        "context_awareness_reason":        "",

        "instruction_total_sections":      4,
        "instruction_complete_sections":   None,
        "instruction_following_score":     None,
        "instruction_following_reason":    "",

        "overall_score":                   None
    }

    for line in raw.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue

        key, _, value = line.partition(":")
        key   = key.strip().lower()
        value = value.strip()

        try:
            # ── Accuracy ──────────────────────────
            if key == "accuracy_total_claims":
                scores["accuracy_total_claims"] = int(value)
            elif key == "accuracy_correct":
                scores["accuracy_correct"] = int(value)
            elif key == "accuracy_incorrect":
                scores["accuracy_incorrect"] = int(value)
            elif key == "accuracy_score":
                scores["accuracy_score"] = float(value)
            elif key == "accuracy_reason":
                scores["accuracy_reason"] = value

            # ── Completeness ──────────────────────
            elif key == "completeness_total_details":
                scores["completeness_total_details"] = int(value)
            elif key == "completeness_captured":
                scores["completeness_captured"] = int(value)
            elif key == "completeness_missing":
                scores["completeness_missing"] = int(value)
            elif key == "completeness_score":
                scores["completeness_score"] = float(value)
            elif key == "completeness_reason":
                scores["completeness_reason"] = value

            # ── Communication ─────────────────────
            elif key == "communication_total_issues":
                scores["communication_total_issues"] = int(value)
            elif key == "communication_quality_score":
                scores["communication_quality_score"] = float(value)
            elif key == "communication_quality_reason":
                scores["communication_quality_reason"] = value

            # ── Context Awareness ─────────────────
            elif key == "context_total_details":
                scores["context_total_details"] = int(value)
            elif key == "context_captured":
                scores["context_captured"] = int(value)
            elif key == "context_missing":
                scores["context_missing"] = int(value)
            elif key == "context_awareness_score":
                scores["context_awareness_score"] = float(value)
            elif key == "context_awareness_reason":
                scores["context_awareness_reason"] = value

            # ── Instruction Following ─────────────
            elif key == "instruction_total_sections":
                scores["instruction_total_sections"] = int(value)
            elif key == "instruction_complete_sections":
                scores["instruction_complete_sections"] = int(value)
            elif key == "instruction_following_score":
                scores["instruction_following_score"] = float(value)
            elif key == "instruction_following_reason":
                scores["instruction_following_reason"] = value

            # ── Overall ───────────────────────────
            elif key == "overall_score":
                scores["overall_score"] = float(value)

        except ValueError:
            continue

    # Calculate overall if LLM missed it
    metric_scores = [
        scores["accuracy_score"],
        scores["completeness_score"],
        scores["communication_quality_score"],
        scores["context_awareness_score"],
        scores["instruction_following_score"]
    ]
    valid = [s for s in metric_scores if s is not None]
    if scores["overall_score"] is None and valid:
        scores["overall_score"] = round(sum(valid) / len(valid), 4)

    return scores

# ── Main ───────────────────────────────────────────
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(V1_PATH, "r") as f:
        v1_data = json.load(f)

    # Resume from checkpoint if exists
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, "r") as f:
            results = json.load(f)
        done_ids = {r["id"] for r in results}
        print(f"Resuming from checkpoint — {len(done_ids)} samples done.")
    else:
        results  = []
        done_ids = set()

    prompt_template = load_prompt()

    for sample in v1_data:
        i          = sample["id"]
        transcript = sample["transcript"]
        generated  = sample["generated"]

        if i in done_ids:
            continue

        print(f"Processing sample {i:02d}...")

        try:
            raw    = evaluate_soap(prompt_template, transcript, generated)
            scores = parse_output(raw)
            status = "OK"
        except Exception as e:
            print(f"  Error at sample {i}: {e}")
            scores = {
                "accuracy_total_claims":         None,
                "accuracy_correct":              None,
                "accuracy_incorrect":            None,
                "accuracy_score":                None,
                "accuracy_reason":               "",
                "completeness_total_details":    None,
                "completeness_captured":         None,
                "completeness_missing":          None,
                "completeness_score":            None,
                "completeness_reason":           "",
                "communication_total_issues":    None,
                "communication_quality_score":   None,
                "communication_quality_reason":  "",
                "context_total_details":         None,
                "context_captured":              None,
                "context_missing":               None,
                "context_awareness_score":       None,
                "context_awareness_reason":      "",
                "instruction_total_sections":    4,
                "instruction_complete_sections": None,
                "instruction_following_score":   None,
                "instruction_following_reason":  "",
                "overall_score":                 None
            }
            raw    = ""
            status = "ERROR"

        result_entry = {
            "id":         i,
            "transcript": transcript,
            "generated":  generated,
            **scores,
            "status":     status,
            "raw_output": raw
        }

        results.append(result_entry)

        print(f"  Sample {i:02d} → "
              f"Overall: {scores['overall_score']} | "
              f"Acc: {scores['accuracy_score']} | "
              f"Comp: {scores['completeness_score']} | "
              f"Comm: {scores['communication_quality_score']} | "
              f"Ctx: {scores['context_awareness_score']} | "
              f"Inst: {scores['instruction_following_score']}")

        # Save checkpoint after every sample
        with open(CHECKPOINT, "w") as f:
            json.dump(results, f, indent=2)

    # Final save
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Remove checkpoint
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    # ── Summary ───────────────────────────────────
    ok_results = [r for r in results if r["status"] == "OK"]

    def avg(key):
        vals = [r[key] for r in ok_results if r[key] is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    print(f"\n{'─'*50}")
    print(f"AGENT 2 EVALUATION COMPLETE")
    print(f"{'─'*50}")
    print(f"Total Samples            : {len(results)}")
    print(f"Successful               : {len(ok_results)}")
    print(f"Errors                   : {sum(1 for r in results if r['status'] == 'ERROR')}")
    print(f"{'─'*50}")
    print(f"Avg Accuracy             : {avg('accuracy_score')}")
    print(f"Avg Completeness         : {avg('completeness_score')}")
    print(f"Avg Communication Quality: {avg('communication_quality_score')}")
    print(f"Avg Context Awareness    : {avg('context_awareness_score')}")
    print(f"Avg Instruction Following: {avg('instruction_following_score')}")
    print(f"Avg Overall Score        : {avg('overall_score')}")
    print(f"{'─'*50}")
    print(f"Results saved to         : {OUT_PATH}")

if __name__ == "__main__":
    run()