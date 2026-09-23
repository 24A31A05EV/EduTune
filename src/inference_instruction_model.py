from transformers import pipeline


MODEL_NAME = "google/flan-t5-small"


def main():
    print("Loading EduTune instruction model...")

    generator = pipeline(
        "text-generation",
        model=MODEL_NAME
    )

    questions = [
        "What is a stack in data structures?",
        "What is normalization in DBMS?",
        "What is binary search?",
        "What is machine learning?",
        "What is an API?"
    ]

    for question in questions:
        prompt = (
            "You are EduTune, an educational tutor. "
            "Answer the following question clearly and concisely "
            "for a computer science student.\n\n"
            f"Question: {question}\n"
            "Answer:"
        )

        result = generator(
            prompt,
            max_new_tokens=100,
            do_sample=False
        )

        print("\nQuestion:", question)
        print("Answer:", result[0]["generated_text"])


if __name__ == "__main__":
    main()