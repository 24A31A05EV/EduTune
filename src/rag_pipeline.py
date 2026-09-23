import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


KNOWLEDGE_FILE = Path("data/raw/knowledge_base.txt")

INDEX_FILE = Path("data/processed/knowledge.index")
DOCUMENTS_FILE = Path("data/processed/documents.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_documents():
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    documents = [
        section.strip()
        for section in text.split("\n\n")
        if section.strip()
    ]

    return documents


def build_index(documents):
    print("\nLoading embedding model...")

    model = SentenceTransformer(EMBEDDING_MODEL)

    print("Creating embeddings...")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(
        embeddings.astype(np.float32)
    )

    return index, model


def save_index(index, documents):
    INDEX_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    faiss.write_index(
        index,
        str(INDEX_FILE)
    )

    with open(
        DOCUMENTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            documents,
            file,
            indent=2,
            ensure_ascii=False
        )


def search(query, index, documents, model, top_k=3):

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype(np.float32),
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):
        if index_id == -1:
            continue

        results.append(
            {
                "score": float(score),
                "document": documents[index_id]
            }
        )

    return results


def main():

    print("=" * 60)
    print("EDUTUNE RAG PIPELINE")
    print("=" * 60)

    documents = load_documents()

    print(f"\nDocuments loaded: {len(documents)}")

    index, model = build_index(documents)

    save_index(index, documents)

    print("\nRAG index created successfully!")

    print(f"Index: {INDEX_FILE}")
    print(f"Documents: {DOCUMENTS_FILE}")

    query = input(
        "\nAsk an educational question: "
    )

    results = search(
        query,
        index,
        documents,
        model
    )

    print("\n" + "=" * 60)
    print("RETRIEVED KNOWLEDGE")
    print("=" * 60)

    for number, result in enumerate(
        results,
        start=1
    ):
        print(
            f"\n[{number}] Similarity: "
            f"{result['score']:.4f}"
        )

        print(result["document"])

    print("\n" + "=" * 60)
    print("RAG SEARCH COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()