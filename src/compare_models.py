import json
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL = "distilgpt2"
LORA_MODEL = "models/edutune-lora"

INDEX_FILE = Path("data/processed/knowledge.index")
DOCUMENTS_FILE = Path("data/processed/documents.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

QUESTIONS = [
    "What is a stack in data structures?",
    "What is normalization in DBMS?",
    "What is binary search?",
    "What is machine learning?",
    "What is an API?"
]


def generate_answer(model, tokenizer, question, context=None):
    if context:
        prompt = f"""You are EduTune, an educational tutor.

Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""
    else:
        prompt = f"""### Instruction:
{question}

### Response:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )


def retrieve_context(question, index, documents, embedding_model):
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype(np.float32),
        3
    )

    contexts = []

    for index_id in indices[0]:
        if index_id != -1:
            contexts.append(documents[index_id])

    return "\n\n".join(contexts)


def main():
    print("=" * 70)
    print("EDUTUNE - BASE vs LORA vs RAG")
    print("=" * 70)

    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(LORA_MODEL)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base_model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading LoRA model...")
    lora_base = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

    edutune_model = PeftModel.from_pretrained(
        lora_base,
        LORA_MODEL
    )

    edutune_model.config.pad_token_id = tokenizer.pad_token_id

    print("Loading RAG system...")
    index = faiss.read_index(str(INDEX_FILE))

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    results = []

    for number, question in enumerate(QUESTIONS, start=1):

        print("\n" + "=" * 70)
        print(f"QUESTION {number}")
        print("=" * 70)
        print(question)

        print("\n[BASE MODEL]")
        base_answer = generate_answer(
            base_model,
            tokenizer,
            question
        )
        print(base_answer)

        print("\n[EDUTUNE LoRA]")
        lora_answer = generate_answer(
            edutune_model,
            tokenizer,
            question
        )
        print(lora_answer)

        print("\n[RAG]")
        context = retrieve_context(
            question,
            index,
            documents,
            embedding_model
        )

        rag_answer = generate_answer(
            base_model,
            tokenizer,
            question,
            context
        )

        print(rag_answer)

        results.append({
            "question": question,
            "base_model": base_answer,
            "edutune_lora": lora_answer,
            "rag": rag_answer
        })

    output_file = Path("evaluation/model_comparison.json")

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETED")
    print("=" * 70)
    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()