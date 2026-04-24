# 📄 AI Research Assistant (Hybrid RAG)

A deployed AI-powered research assistant that enables users to upload PDFs and perform grounded question answering with source attribution, summarization, and reliability validation.

---

## 🚀 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** system designed to answer questions strictly from user-provided documents.  

Unlike generic LLM tools, the system retrieves relevant document chunks and generates responses grounded in that context, improving reliability and reducing hallucinations.

---

## Features

- **Multi-document PDF ingestion**
- **Hybrid retrieval (Dense + BM25)**
- **Context-grounded Q&A**
- **Source attribution (document + page)**
- **Summarization of full documents**
- **LLM-based answer validation (reliability layer)**
- **Retrieval confidence scoring**
- **Latency tracking (retrieval, generation, total)**

---

## System Architecture


---

## How It Works

1. **Document Ingestion**
   - PDFs are parsed and split into chunks
   - Chunks are embedded using Gemini embeddings
   - Stored in ChromaDB for vector similarity search

2. **Hybrid Retrieval**
   - Combines:
     - Dense embeddings (semantic search)
     - BM25 (keyword search)
   - Improves retrieval accuracy for both semantic and exact queries

3. **Grounded Generation**
   - LLM generates answers using only retrieved context
   - Prevents open-ended hallucinations

4. **Reliability Layer**
   - Second LLM pass validates if answer is supported by context
   - Low-confidence answers are flagged or rejected

5. **Summarization**
   - Allows users to extract key insights from entire documents

---

## Performance Metrics

- **Retrieval Latency:** ~111 ms (avg)
- **End-to-End Latency:** ~6.4 s (avg, includes validation)
- **Confidence Scoring:** Based on retrieval similarity
- **Validation:** Ensures answers are context-grounded

> Note: Latency is dominated by LLM generation + validation calls.

---

## Tech Stack

**AI / LLM**
- Gemini API (generation + embeddings)
- RAG (Retrieval-Augmented Generation)

**Retrieval**
- ChromaDB (vector database)
- BM25 (keyword-based retrieval)

**Backend**
- Python
- FastAPI (modular service design)

**Frontend**
- Streamlit (deployed UI)

**Data Processing**
- PyMuPDF (PDF parsing)

---

## Deployment

- Hosted on **Streamlit Cloud**
- Public, interactive interface for document upload and querying

---

## ▶️ How to Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/your-username/ai-research-assistant.git
cd ai-research-assistant

pip install -r requirements.txt

export GOOGLE_API_KEY=your_api_key

streamlit run streamlit_app.py