import ollama

def load_prompt(prompt_path):
    with open(prompt_path, "r") as f:
        return f.read()

def _render_prompt_optimizations(prompt_optimizations_list: list[str] | None) -> str:
    items = prompt_optimizations_list or []
    if not items:
        return "None."
    return "\n".join(f"- {x}" for x in items)


def generate_soap_v1(transcript, prompt_path="v1/Few_shot.txt", prompt_optimizations_list=[]):
    prompt_template = load_prompt(prompt_path)
    prompt_optimizations = _render_prompt_optimizations(prompt_optimizations_list)
    prompt = prompt_template.format(transcript=transcript, prompt_optimizations=prompt_optimizations)

    response = ollama.chat(
        model="mistral",  # or llama3
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()