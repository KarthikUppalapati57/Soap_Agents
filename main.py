from pipeline.pipeline import Pipeline
import json

def main():
    pipeline = Pipeline(data_path="data/", prompt_folder="prompts")
    stream =pipeline.run()

    for data in stream:
        json.dump(data, open("Output/claim_verification.json", "w"), indent=4)

if __name__ == "__main__":
    main()
