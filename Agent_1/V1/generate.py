import ollama

def load_prompt(prompt_path="v1/prompt.txt"):
    with open(prompt_path, "r") as f:
        return f.read()

def generate_soap_v1(transcript, prompt_path="v1/prompt.txt"):
    prompt_template = load_prompt(prompt_path)
    prompt = prompt_template.replace("{transcript}", transcript)

    response = ollama.chat(
        model="mistral",  # or llama3
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()