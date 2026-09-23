from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "distilgpt2"


def main():
    print("Loading EduTune base model...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    question = input("\nAsk EduTune a question: ")

    prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    print("\nGenerating answer...\n")

    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    answer = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    print(answer)


if __name__ == "__main__":
    main()