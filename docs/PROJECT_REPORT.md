# EduTune — Domain-Specific Educational AI Tutor

## 1. Project Overview

EduTune is an AI-powered educational tutor designed to provide students with concise and topic-focused explanations across common computer science subjects.

The project explores three approaches to educational question answering:

1. Base language model
2. LoRA fine-tuned language model
3. Retrieval-based educational knowledge system

The project also provides a FastAPI backend and a Streamlit user interface so that the system can be accessed as a complete application.

---

## 2. Problem Statement

Students often need quick explanations of technical concepts while learning subjects such as Data Structures, DBMS, Operating Systems, Python, Machine Learning, APIs, and GitHub.

General-purpose language models may produce repetitive or inaccurate responses for domain-specific educational questions.

EduTune explores how dataset preparation, parameter-efficient fine-tuning, and retrieval-based methods can be used to create a more focused educational assistant.

---

## 3. Objectives

The main objectives of EduTune are:

* Build a domain-specific educational dataset.
* Prepare and validate training data.
* Experiment with LoRA-based fine-tuning.
* Compare a base model with a fine-tuned model.
* Build a retrieval-based educational baseline.
* Create an API for educational question answering.
* Develop a simple student-friendly web interface.
* Evaluate the system using representative educational questions.
* Maintain the complete project as a reproducible GitHub repository.

---

## 4. Technology Stack

| Component            | Technology            |
| -------------------- | --------------------- |
| Programming Language | Python                |
| Language Model       | DistilGPT-2           |
| Fine-Tuning          | LoRA / PEFT           |
| Training Framework   | Hugging Face TRL      |
| Dataset Processing   | Hugging Face Datasets |
| Embeddings           | Sentence Transformers |
| Vector Search        | FAISS                 |
| Backend              | FastAPI               |
| Frontend             | Streamlit             |
| Data Processing      | Pandas                |
| Evaluation           | Python / Scikit-learn |
| Version Control      | Git / GitHub          |

---

## 5. System Architecture

```text
                    Student Question
                           |
                           v
                  +------------------+
                  | Streamlit UI     |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | FastAPI Backend  |
                  +--------+---------+
                           |
                           v
                  +------------------+
                  | Question         |
                  | Processing       |
                  +--------+---------+
                           |
             +-------------+-------------+
             |                           |
             v                           v
     +---------------+          +----------------+
     | LoRA Model    |          | FAISS RAG      |
     | Experiment    |          | Retrieval      |
     +---------------+          +----------------+
             |                           |
             +-------------+-------------+
                           |
                           v
                  Educational Answer
```

---

## 6. Dataset Preparation

The initial EduTune dataset contains educational examples covering topics including:

* Data Structures
* DBMS
* Operating Systems
* Python
* Machine Learning
* APIs
* GitHub

Each example contains:

```text
instruction
input
output
```

The dataset preparation script validates these fields and converts the examples into a format suitable for language-model training.

The processed dataset is divided into:

* Training dataset
* Validation dataset

---

## 7. LoRA Fine-Tuning

EduTune uses Parameter-Efficient Fine-Tuning (PEFT) with LoRA.

Instead of updating every parameter of the language model, LoRA introduces trainable low-rank matrices into selected model components.

This reduces the computational requirements compared with full model fine-tuning.

The experimental model used in the project is:

```text
DistilGPT-2
```

The fine-tuned artifacts are generated locally and are excluded from GitHub through `.gitignore`.

---

## 8. Retrieval-Based Educational Assistant

EduTune also implements a retrieval-based baseline.

The educational knowledge base is converted into vector embeddings using Sentence Transformers.

FAISS is then used to perform similarity-based retrieval.

The process is:

```text
Question
   |
   v
Sentence Transformer
   |
   v
Question Embedding
   |
   v
FAISS Similarity Search
   |
   v
Relevant Educational Document
```

This approach provides a useful baseline for comparing retrieval with language-model-based approaches.

---

## 9. API Layer

The FastAPI application provides endpoints for interacting with EduTune.

### Root Endpoint

```text
GET /
```

Returns the API status and project information.

### Health Endpoint

```text
GET /health
```

Used to verify that the API is running correctly.

### Question Endpoint

```text
POST /ask
```

Accepts a student question and returns the retrieved educational answer and similarity information.

---

## 10. Streamlit Interface

The Streamlit application provides a student-friendly interface for interacting with EduTune.

The interface includes:

* Example questions
* Custom question input
* Ask button
* Educational response
* Similarity information
* Technology-stack information

The interface communicates with the FastAPI backend.

---

## 11. Evaluation

EduTune uses representative educational questions to examine the behavior of the different approaches.

Example topics include:

* Stack
* Binary Search
* Normalization
* Machine Learning
* APIs

The evaluation is intended to demonstrate the differences between:

```text
Base Model
     vs
LoRA Fine-Tuned Model
     vs
Retrieval-Based Baseline
```

The current implementation is an experimental project rather than a production-grade educational AI system.

---

## 12. Limitations

The current version has several limitations:

* The training dataset is relatively small.
* The local environment uses CPU-only PyTorch.
* DistilGPT-2 is not specifically optimized for instruction following.
* Fine-tuning results may therefore be limited.
* The retrieval system depends on the quality and coverage of the knowledge base.
* The current retrieval baseline focuses on retrieving relevant educational content rather than providing sophisticated generated RAG responses.

These limitations provide opportunities for future improvements.

---

## 13. Future Improvements

Future versions of EduTune can include:

* A larger educational dataset.
* A stronger instruction-tuned language model.
* QLoRA experimentation on a suitable GPU environment.
* Improved evaluation metrics.
* More comprehensive RAG generation.
* Automatic citation of retrieved educational sources.
* Conversation history.
* Student-level personalization.
* Topic classification.
* Difficulty-level adaptation.
* Authentication and user accounts.
* Cloud deployment.
* Monitoring and logging.

---

## 14. Learning Outcomes

Through this project, the following concepts are explored:

* Dataset preparation
* Natural Language Processing
* Transformer models
* Parameter-Efficient Fine-Tuning
* LoRA
* Vector embeddings
* Semantic search
* FAISS
* Retrieval-Augmented Generation concepts
* REST APIs
* FastAPI
* Streamlit
* Model evaluation
* Git and GitHub
* End-to-end AI application development

---

## 15. Conclusion

EduTune demonstrates an end-to-end approach to building an educational AI application.

The project combines machine learning experimentation, retrieval-based information access, API development, and a user-facing web interface into a single repository.

The primary goal is to understand the complete workflow involved in developing a domain-specific AI assistant, from dataset preparation and model experimentation to deployment-oriented application development.
