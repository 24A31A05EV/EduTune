from pathlib import Path
import json

import faiss
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INDEX_FILE = Path("data/processed/knowledge.index")
DOCUMENTS_FILE = Path("data/processed/documents.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="EduTune API",
    description="Educational AI Tutor API using Retrieval-Augmented Generation",
    version="1.0.0"
)


# --------------------------------------------------
# Request model
# --------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


# --------------------------------------------------
# Load RAG components
# --------------------------------------------------

print("Loading EduTune RAG system...")

index = faiss.read_index(str(INDEX_FILE))

with open(
    DOCUMENTS_FILE,
    "r",
    encoding="utf-8"
) as file:
    documents = json.load(file)

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("EduTune RAG system loaded.")


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def home():

    return {
        "project": "EduTune",
        "status": "running",
        "message": "Educational AI Tutor API"
    }


# --------------------------------------------------
# Health check
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Ask EduTune
# --------------------------------------------------

@app.post("/ask")
def ask_edutune(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        return {
            "error": "Question cannot be empty."
        }

    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    scores, indices = index.search(
        query_embedding.astype(np.float32),
        1
    )

    index_id = int(indices[0][0])
    similarity = float(scores[0][0])

    if index_id == -1:
        return {
            "question": question,
            "answer": "I could not find relevant educational information.",
            "similarity": 0
        }

    answer = documents[index_id]

    return {
        "question": question,
        "answer": answer,
        "similarity": round(similarity, 4)
    }