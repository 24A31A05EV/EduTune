import json
from pathlib import Path
from sklearn.model_selection import train_test_split


RAW_FILE = Path("data/raw/edutune_dataset.jsonl")
PROCESSED_DIR = Path("data/processed")

TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "validation.jsonl"


def load_dataset():
    examples = []

    with open(RAW_FILE, "r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {error}"
                )

            required_fields = ["instruction", "input", "output"]

            for field in required_fields:
                if field not in example:
                    raise ValueError(
                        f"Missing '{field}' on line {line_number}"
                    )

            examples.append(example)

    return examples


def format_example(example):
    instruction = example["instruction"]
    user_input = example["input"].strip()
    output = example["output"]

    if user_input:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{user_input}\n\n"
            f"### Response:\n{output}"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )

    return {"text": prompt}


def save_jsonl(data, filepath):
    with open(filepath, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    print("Loading EduTune dataset...")

    examples = load_dataset()

    print(f"Loaded examples: {len(examples)}")

    formatted_examples = [
        format_example(example)
        for example in examples
    ]

    train_data, validation_data = train_test_split(
        formatted_examples,
        test_size=0.2,
        random_state=42
    )

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    save_jsonl(train_data, TRAIN_FILE)
    save_jsonl(validation_data, VAL_FILE)

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(validation_data)}")

    print(f"Saved training data: {TRAIN_FILE}")
    print(f"Saved validation data: {VAL_FILE}")

    print("\nDataset preparation completed successfully!")


if __name__ == "__main__":
    main()