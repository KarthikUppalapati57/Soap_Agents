from pipeline.agent_interface import AgentInterface
import json
import os

class Pipeline:
    def __init__(self, data_path: str, prompt_folder: str, output_dir: str = "Output", limit: int = 5):
        self.data_path = data_path
        self.prompt_folder = prompt_folder
        self.output_dir = output_dir
        self.agent_interface = AgentInterface(prompt_folder=self.prompt_folder)
        self.limit = limit
        self.count = 0

    def _save_item_output(
        self,
        index: int,
        item: dict,
        claim_result: dict,
        prompt_optimizations_list: list,
        iterations: int,
    ) -> str:
        """Write this row to ``{output_dir}/item_{index:04d}/`` and return that path."""
        sub = os.path.join(self.output_dir, f"item_{index:04d}")
        os.makedirs(sub, exist_ok=True)

        def _dump(name: str, obj) -> None:
            path = os.path.join(sub, name)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

        _dump("meta.json", {"index": index, "iterations": iterations})
        _dump(
            "source.json",
            {
                "transcript": item.get("transcript"),
                "ground_truth": item.get("ground_truth"),
            },
        )
        _dump("agent3_result.json", claim_result)
        _dump("prompt_optimizations.json", prompt_optimizations_list)
        return sub

    def run(self):
        data_path = os.path.join(self.data_path, "clean_medsynth_final.json")
        with open(data_path, encoding="utf-8") as f:
            data = json.load(f)
        os.makedirs(self.output_dir, exist_ok=True)

        for index, item in enumerate(data):
            if self.count >= self.limit:
                break
            self.count += 1
            iteration = 0
            prompt_optimizations_list = []
            while True:
                iteration += 1
                transcript = item["transcript"]
                claim_verification = self.agent_interface.run(transcript, prompt_optimizations_list)

                unsupported_claims = []
                claims = (claim_verification.get("claim_verification") or {}).get("claims") or []
                for claim in claims:
                    if claim.get("support_status") == "unsupported":
                        unsupported_claims.append(claim)

                if len(unsupported_claims) == 0 or iteration > 3:
                    out_dir = self._save_item_output(
                        index,
                        item,
                        claim_verification,
                        prompt_optimizations_list,
                        iteration,
                    )
                    print(f"Saved: {out_dir}")
                    yield claim_verification, prompt_optimizations_list
                    break

                prompt_optimizations = self.agent_interface.prompt_optimizer(
                    transcript, unsupported_claims, prompt_optimizations_list
                )
                prompt_optimizations_list.append(prompt_optimizations)
                print(f"Prompt Optimizations: {prompt_optimizations}")

