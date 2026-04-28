import ollama

def load_prompt(prompt_path="v1/prompt.txt"):
    with open(prompt_path, "r") as f:
        return f.read()

def _optimizations_to_str(prompt_optimizations) -> str:
    if prompt_optimizations is None:
        return ""
    if isinstance(prompt_optimizations, list):
        return "\n".join(str(x) for x in prompt_optimizations)
    return str(prompt_optimizations)


def generate_soap_v1(transcript, prompt_path="v1/prompt.txt", prompt_optimizations=None):
    prompt_template = load_prompt(prompt_path)
    prompt = prompt_template.replace("{transcript}", transcript)
    prompt = prompt.replace(
        "{prompt_optimizations}", _optimizations_to_str(prompt_optimizations)
    )


    response = ollama.chat(
        model="mistral",  # or llama3
        messages=[{"role": "user", "content": prompt}],
        options={
            "temperature": 0.0
        }
    )

    return response["message"]["content"].strip()