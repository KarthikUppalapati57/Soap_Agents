import json
import os
from v1.generate import generate_soap_v1

def run_v1(num_samples=50):

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))

    file_path = os.path.join(BASE_DIR, "data", "clean_medsynth_final.json")

    print("Dataset path:", file_path)  # debug

    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for i in range(num_samples):
        sample = data[i]

        transcript = sample["transcript"]
        ground_truth = sample["ground_truth"]

        try:
            generated = generate_soap_v1(transcript)
        except Exception as e:
            print(f"Error at sample {i}: {e}")
            continue

        results.append({
            "id": i,
            "transcript": transcript,
            "generated": generated,
            "ground_truth": ground_truth
        })

        print(f"Processed sample {i}")

    with open("v1_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_v1(50)