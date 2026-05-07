from pipeline.pipeline import Pipeline
import argparse
from dotenv import load_dotenv

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    pipeline = Pipeline(data_path="data/", prompt_folder="prompts", output_dir="Output", limit=args.limit)
    for _claim_verification, _prompt_opts in pipeline.run():
        pass  

if __name__ == "__main__":
    main()
