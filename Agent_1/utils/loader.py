import json

def load_clean_medsynth(path="data/clean_medsynth_final.json"):
    """
    Loads the cleaned MedSynth dataset from JSON file.

    Returns:
        list of dicts with keys:
        - transcript
        - ground_truth
    """

    with open(path, "r") as f:
        dataset = json.load(f)

    return dataset