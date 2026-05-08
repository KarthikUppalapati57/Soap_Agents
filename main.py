from pipeline.pipeline import Pipeline
import argparse
from dotenv import load_dotenv

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="Output")
    parser.add_argument("--only_generate", action="store_true", default=False)
    parser.add_argument("--no_mkg", action="store_true", default=False)
    args = parser.parse_args()

    pipeline = Pipeline(data_path="data/", prompt_folder="prompts", output_dir=args.output_dir, limit=args.limit)
    for _claim_verification, _prompt_opts in pipeline.run(only_generate=args.only_generate, no_mkg=args.no_mkg):
        pass  

if __name__ == "__main__":
    main()
