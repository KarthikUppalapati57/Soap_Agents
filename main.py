from pipeline.pipeline import Pipeline


def main():

    pipeline = Pipeline(data_path="data/", prompt_folder="prompts", output_dir="Output")
    for _claim_verification, _prompt_opts in pipeline.run():
        pass  

if __name__ == "__main__":
    main()
