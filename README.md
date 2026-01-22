# 📚 RAG Study Bot (Local)

A Retrieval-Augmented Generation (RAG) application that allows users to upload
PDF, DOCX, or PPTX documents and ask questions based strictly on their content.

Built using:
- LangChain (stable 0.1.x)
- FAISS vector store
- HuggingFace FLAN-T5
- Streamlit UI
- No OpenAI / API keys required

---

## 🚀 Features

- Multi-document upload
- Conversational Q&A
- Source citations
- Local inference (privacy-friendly)
- Interview & demo ready

---

## 🧰 Tech Stack

- Python 3.10 / 3.11
- LangChain 0.1.20
- FAISS
- SentenceTransformers
- HuggingFace Transformers
- Streamlit

---

## ⚙️ Installation

```bash
python -m venv rag_env
source rag_env/bin/activate   # Windows: rag_env\Scripts\activate

pip install -r requirements.txt
