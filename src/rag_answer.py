import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


INDEX_FILE = Path("data/processed/knowledge.index")
DOCUMENTS_FILE = Path("data/processed/documents.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_rag():
    index = faiss.read_index(str(INDEX_FILE))

    with open(DOCUMENTS_FILE, "r", encoding="utf-8") as file:
        documents = json.load(file)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    return index, documents, embedding_model


def retrieve(query, index, documents, embedding_model, top_k=1):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype(np.float32),
        top_k
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):

        if index_id == -1:
            continue

        results.append({
            "score": float(score),
            "document": documents[index_id]
        })

    return results


def main():

    print("=" * 60)
    print("EDUTUNE RAG BASELINE")
    print("=" * 60)

    index, documents, embedding_model = load_rag()

    question = input("\nAsk EduTune a question: ")

    results = retrieve(
        question,
        index,
        documents,
        embedding_model
    )

    print("\n" + "=" * 60)
    print("EDUTUNE RAG ANSWER")
    print("=" * 60)

    if not results:
        print("No relevant information found.")
        return

    best_result = results[0]

    print(f"\nSimilarity: {best_result['score']:.4f}")
    print()
    print(best_result["document"])

    print("\n" + "=" * 60)
    print("RAG BASELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()