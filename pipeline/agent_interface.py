from Agent_1.V1.generate import generate_soap_v1
from Agent_2.v2.OpenAI_health_Benchmark.agent2 import evaluate_soap, parse_output
from Agent_2.v2.MKG.mkg_validation import extract_terms, validate_terms
from Agent_3.MKG import PrimeKGExplorer, extract_medical_terms
from Agent_3.agent3 import _process_one
from google.genai import Client, types
import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()



class AgentInterface:
    def __init__(self, prompt_folder="prompts"):
        self.agent_1 = generate_soap_v1
        self.agent_2 = evaluate_soap, parse_output
        self.agent_3 = _process_one
        self.prompt_folder = prompt_folder
        self.client = Client()
        self.model = os.getenv("AGENT3_GEMINI_MODEL")
        self.mkg_client = None
        try:
            # PrimeKGExplorer defaults to `Agent_3/mkg/kg.csv` (or PRIMEKG_CSV if set)
            self.mkg_client = PrimeKGExplorer()
        except Exception:
            self.mkg_client = None

    def _max_tokens(self) -> int | None:
        v = (os.getenv("MAX_TOKENS") or "2048").strip()
        if not v:
            return None
        try:
            n = int(v)
        except ValueError:
            return None
        return n if n > 0 else None

    def _primekg_context(self, generated_soap: str, parsed_output: dict, *, max_rows: int = 40) -> str:
        if not self.mkg_client:
            return "PrimeKG context unavailable (missing PRIMEKG_CSV / kg.csv)."

        hops = int(os.getenv("MKG_KG_HOPS", "1"))
        # Pool of raw triple rows to pull from the graph (before deduplicating to prompt lines)
        max_pool = int(os.getenv("MKG_KG_MAX_ROWS", "2000"))
        max_gliner_terms = int(os.getenv("MKG_GLINER_MAX_TERMS", "12"))
        max_query_terms = int(os.getenv("MKG_PRIMEKG_QUERY_TERMS", "8"))
        gliner_threshold = float(os.getenv("MKG_GLINER_THRESHOLD", "0.35"))

        # Primary: GLiNER medical terms from the generated SOAP (Agent_3.MKG.extract_medical_terms).
        candidates: list[str] = []
        try:
            for t in extract_medical_terms(
                generated_soap,
                threshold=gliner_threshold,
            )[:max_gliner_terms]:
                if len(t) >= 2:
                    candidates.append(t)
        except Exception:
            candidates = []

        if not candidates:
            # Fallback if GLiNER is unavailable, returns nothing, or all terms too short.
            for k in ("diagnosis", "primary_diagnosis", "condition", "disease", "problem"):
                v = parsed_output.get(k)
                if isinstance(v, str) and v.strip() and len(v.strip()) >= 2:
                    candidates.append(v.strip())
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.strip() and len(item.strip()) >= 2:
                            candidates.append(item.strip())
            if not candidates:
                candidates = [generated_soap[:2000]]

        # Query: graph-expanded related triples per candidate, then merge unique lines for the prompt.
        out_lines: list[str] = []
        seen: set[tuple[str, str, str]] = set()
        for c in candidates[:max_query_terms]:
            try:
                df = self.mkg_client.get_related_triples(c, hops=hops, max_rows=max_pool)
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            for _, row in df.iterrows():
                x = str(row.get("x_name", "")).strip()
                rel = str(row.get("relation", "")).strip()
                y = str(row.get("y_name", "")).strip()
                if not (x and rel and y):
                    continue
                key = (x, rel, y)
                if key in seen:
                    continue
                seen.add(key)
                out_lines.append(f"- {x} — {rel} — {y}")
                if len(out_lines) >= max_rows:
                    break
            if len(out_lines) >= max_rows:
                break

        return "\n".join(out_lines) if out_lines else "No PrimeKG matches found for extracted medical terms."

    def run_agent_1(self, transcript: str, prompt_optimizations_list: list) -> str:
        prompt_path = os.path.join(self.prompt_folder, "A1_Few_shot.txt")
        return self.agent_1(transcript, prompt_path, prompt_optimizations_list)

    def run_agent_2(self, transcript: str, generated: str) -> str:
        prompt_path = os.path.join(self.prompt_folder, "A2_prompt.txt")
        prompt_template = open(prompt_path, "r").read()
        evaluation = self.agent_2[0](transcript=transcript, generated=generated, prompt_template=prompt_template)
        parsed_output = self.agent_2[1](evaluation)
        parsed_output["transcript"] = transcript
        parsed_output["generated"] = generated

        terms = extract_terms(generated)
        valid_terms, invalid_terms = validate_terms(terms)
        parsed_output["UMLS_valid_terms"] = valid_terms
        parsed_output["UMLS_invalid_terms"] = invalid_terms
        parsed_output["UMLS_accuracy_score"] = len(valid_terms) / (len(valid_terms) + len(invalid_terms))

        return parsed_output

    def run_agent_3(self, parsed_output: dict, mkg_terms: str, client: Client) -> str:
        return self.agent_3(parsed_output, medical_knowledge_terms=mkg_terms, client=client)

    def run(self, transcript: str, prompt_optimizations_list: list, only_generate: bool = False, no_mkg: bool = False) -> str:
        print("Running Agent 1")
        generated_soap = self.run_agent_1(transcript, prompt_optimizations_list)
        print("Running Agent 2")
        parsed_output = self.run_agent_2(transcript, generated_soap)
        print("Running Agent 3")
        if not no_mkg:
            mkg_context_text = self._primekg_context(generated_soap, parsed_output)
            print(f"MKG Context Text: {mkg_context_text.strip()[:100]}", file=sys.stderr)
        else:
            mkg_context_text = ""
        claim_verification = self.run_agent_3(parsed_output, mkg_context_text, client=self.client)

        return claim_verification, mkg_context_text

    def prompt_optimizer(self, transcript: str, unsupported_claims: list, prompt_optimizations_list: list, mkg_context_text: str) -> str:
        prompt_path = os.path.join(self.prompt_folder, "A3_optimizer_prompt.txt")
        prompt_template = open(prompt_path, "r").read()
        prompt = prompt_template.format(transcript=transcript, unsupported_claims=unsupported_claims, previous_optimizations=prompt_optimizations_list, mkg_context_text=mkg_context_text)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="text/plain",
                temperature=0.0,
                max_output_tokens=self._max_tokens(),
            )
        )
        return response.text

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