import ollama

def load_prompt():
    with open("v1/prompt.txt", "r") as f:
        return f.read()

def generate_soap_v1(transcript):
    prompt_template = load_prompt()
    prompt = prompt_template.replace("{transcript}", transcript)

    response = ollama.chat(
        model="mistral",  # or llama3
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"].strip()