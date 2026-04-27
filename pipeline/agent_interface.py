from Agent_1.V1.generate import generate_soap_v1
from Agent_2.v2.OpenAI_health_Benchmark.agent2 import evaluate_soap, parse_output, load_prompt
from Agent_3.agent3 import _process_one
import pandas as pd
import os
import json
import sys

class AgentInterface:
    def __init__(self, prompt_folder="prompts"):
        self.agent_1 = generate_soap_v1
        self.agent_2 = evaluate_soap, parse_output
        self.agent_3 = _process_one
        self.prompt_folder = prompt_folder

    def run_agent_1(self, transcript: str) -> str:
        return self.agent_1(transcript, os.path.join(self.prompt_folder, "A1_prompt.txt"))

    def run_agent_2(self, transcript: str, generated: str) -> str:
        prompt_path = os.path.join(self.prompt_folder, "A2_prompt.txt")
        prompt_template = open(prompt_path, "r").read()
        evaluation = self.agent_2[0](transcript=transcript, generated=generated, prompt_template=prompt_template)
        parsed_output = self.agent_2[1](evaluation)
        parsed_output["transcript"] = transcript
        parsed_output["generated"] = generated
        return parsed_output

    def run_agent_3(self, parsed_output: dict) -> str:
        return self.agent_3(parsed_output)

    def run(self, transcript: str) -> str:
        generated_soap = self.run_agent_1(transcript)
        parsed_output = self.run_agent_2(transcript, generated_soap)
        claim_verification = self.run_agent_3(parsed_output)
        return claim_verification

if __name__ == "__main__":
    # data "data/clean_medsynth_final.json"
    data = json.load(open("data/clean_medsynth_final.json"))

    for index, item in enumerate(data):
        transcript = item["transcript"]
        ground_truth = item["ground_truth"]
        claim_verification = AgentInterface().run(transcript)
        print(claim_verification)
        if index % 3 == 0:
            print(f"Processed {index} rows")
            break