import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/processed/knowledge.index")
DOCUMENTS_FILE = Path("data/processed/documents.json")
OUTPUT_FILE = Path("evaluation/rag_evaluation.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


QUESTIONS = [
    {
        "question": "What is a stack in data structures?",
        "expected_topic": "stack"
    },
    {
        "question": "What is normalization in DBMS?",
        "expected_topic": "normalization"
    },
    {
        "question": "What is binary search?",
        "expected_topic": "binary search"
    },
    {
        "question": "What is machine learning?",
        "expected_topic": "machine learning"
    },
    {
        "question": "What is an API?",
        "expected_topic": "api"
    }
]


def load_rag():

    index = faiss.read_index(str(INDEX_FILE))

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    model = SentenceTransformer(EMBEDDING_MODEL)

    return index, documents, model


def retrieve(
    question,
    index,
    documents,
    model,
    top_k=1
):

    embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        embedding.astype(np.float32),
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id == -1:
            continue

        results.append({
            "score": float(score),
            "document": documents[index_id]
        })

    return results


def main():

    print("=" * 60)
    print("EDUTUNE RAG EVALUATION")
    print("=" * 60)

    index, documents, model = load_rag()

    results = []

    for item in QUESTIONS:

        question = item["question"]
        expected_topic = item["expected_topic"]

        retrieved = retrieve(
            question,
            index,
            documents,
            model
        )

        if not retrieved:
            results.append({
                "question": question,
                "expected_topic": expected_topic,
                "retrieved": False,
                "similarity": 0
            })
            continue

        best = retrieved[0]

        document = best["document"]

        topic_found = (
            expected_topic.lower()
            in document.lower()
        )

        results.append({
            "question": question,
            "expected_topic": expected_topic,
            "retrieved": True,
            "topic_match": topic_found,
            "similarity": round(
                best["score"],
                4
            ),
            "document": document
        })

        print("\nQuestion:")
        print(question)

        print("\nSimilarity:")
        print(f"{best['score']:.4f}")

        print("\nTopic Match:")
        print(topic_found)

        print("-" * 60)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False
        )

    successful = sum(
        1
        for result in results
        if result.get("topic_match", False)
    )

    total = len(results)

    accuracy = successful / total * 100

    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    print(f"Questions tested: {total}")
    print(f"Correct retrievals: {successful}")
    print(f"Retrieval accuracy: {accuracy:.2f}%")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()