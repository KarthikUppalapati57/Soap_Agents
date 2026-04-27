from pipeline.agent_interface import AgentInterface
import json
import os

class Pipeline:
    def __init__(self, data_path: str, prompt_folder: str):
        self.data_path = data_path
        self.prompt_folder = prompt_folder
        self.agent_interface = AgentInterface(prompt_folder=self.prompt_folder)

    def run(self) -> str:
        data = json.load(open(os.path.join(self.data_path, "clean_medsynth_final.json")))
        processed_data = []
        for index, item in enumerate(data):
            valid = False

            while not valid:
                transcript = item["transcript"]
                ground_truth = item["ground_truth"]
                claim_verification = self.agent_interface.run(transcript)
                accuracy = claim_verification["accuracy_score"]
                print(f"Accuracy: {accuracy}")
                if round(accuracy, 2) >= 0.9:
                    valid = True
                    processed_data.append(claim_verification)
                    break
            yield processed_data

