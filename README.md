# 🎓 EduTune — Domain-Specific Educational AI Tutor

EduTune is a domain-specific educational AI tutor designed to help students understand computer science concepts through an AI-powered question-answering system.

The project combines **LLM fine-tuning, LoRA, Retrieval-Augmented Generation (RAG), FAISS, FastAPI, and Streamlit** into an end-to-end educational AI application.

---

## 📌 Problem Statement

Students often have difficulty understanding technical subjects such as:

- Data Structures
- DBMS
- Operating Systems
- Python
- Machine Learning
- APIs
- Git and GitHub

General-purpose language models may provide broad answers, but they may also produce repetitive, inaccurate, or poorly structured responses.

EduTune explores two approaches for building a domain-specific educational assistant:

1. **LoRA fine-tuning**
2. **Retrieval-Augmented Generation (RAG)**

The project evaluates these approaches against a base language model.

---

## 🎯 Objectives

- Build a domain-specific educational dataset.
- Prepare training and validation data.
- Fine-tune a language model using LoRA.
- Build a RAG-based educational knowledge system.
- Compare base, fine-tuned, and RAG approaches.
- Create a REST API using FastAPI.
- Build an interactive interface using Streamlit.
- Produce reproducible evaluation results.
- Document the complete ML workflow.

---

## 🏗️ System Architecture

```text
                         ┌──────────────────────┐
                         │       Student        │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │    Streamlit UI      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FastAPI         │
                         │       /ask           │
                         └──────────┬───────────┘
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                         ▼                     ▼
                ┌─────────────────┐   ┌─────────────────┐
                │  RAG Retrieval  │   │  LoRA Model     │
                │ FAISS + MiniLM  │   │ Fine-tuning     │
                └────────┬────────┘   └─────────────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Educational     │
                │ Knowledge Base  │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Educational     │
                │ Answer          │
                └─────────────────┘