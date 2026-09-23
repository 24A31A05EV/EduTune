import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


QUESTIONS = [
    "What is a stack in data structures?",
    "What is normalization in DBMS?",
    "What is binary search?",
    "What is machine learning?",
    "What is an API?"
]


def run_instruction_model():
    print("Loading FLAN-T5 instruction model...")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    results = []

    for question in QUESTIONS:
        prompt = (
            "You are EduTune, an educational tutor. "
            "Answer clearly and concisely for a computer science student.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        inputs = tokenizer(prompt, return_tensors="pt")

        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        results.append({
            "question": question,
            "answer": answer
        })

    return results


def save_results(results):
    output_path = Path("evaluation/instruction_model_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    results = run_instruction_model()

    save_results(results)

    print("\n===== EduTune Instruction Model Results =====")

    for item in results:
        print("\nQuestion:", item["question"])
        print("Answer:", item["answer"])