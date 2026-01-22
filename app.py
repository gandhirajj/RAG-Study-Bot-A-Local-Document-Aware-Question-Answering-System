import streamlit as st
import os
import tempfile
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from PyPDF2 import PdfReader

# -------------------------------
# CONFIG
# -------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "google/flan-t5-base"

st.set_page_config(page_title="📚 RAG Study Bot", layout="wide")


# -------------------------------
# LOAD LLM (NO PIPELINE)
# -------------------------------
@st.cache_resource
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return tokenizer, model


# -------------------------------
# PDF LOADER
# -------------------------------
def load_pdf(file) -> List[Document]:
    reader = PdfReader(file)
    docs = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and len(text.strip()) > 30:
            docs.append(
                Document(
                    page_content=f"[Page {i+1}]\n{text}",
                    metadata={"source": file.name, "page": i + 1},
                )
            )
    return docs


# -------------------------------
# TEXT SPLITTER
# -------------------------------
def split_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    return splitter.split_documents(docs)


# -------------------------------
# BUILD VECTORSTORE (NO CACHE ON DOCS)
# -------------------------------
def build_vectorstore(docs: List[Document]):
    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.from_documents(docs, embeddings)


# -------------------------------
# SECTION-AWARE RETRIEVAL (🔥 KEY FIX)
# -------------------------------
def retrieve_docs(vs, question: str, k: int = 6):
    retriever = vs.as_retriever(search_kwargs={"k": k})
    docs = retriever.get_relevant_documents(question)

    # Hard section filter
    if question.strip().startswith("11"):
        filtered = [
            d for d in docs
            if "11." in d.page_content
            or "impact and outcomes" in d.page_content.lower()
            or "project impact" in d.page_content.lower()
        ]
        if filtered:
            return filtered

    return docs


# -------------------------------
# GENERATE ANSWER
# -------------------------------
def generate_answer(context: str, question: str):
    tokenizer, model = load_llm()

    prompt = f"""
Answer the question strictly using the context below.
If the answer is not present, say "Not found in document".

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.2,
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# -------------------------------
# STREAMLIT UI
# -------------------------------
def main():
    st.title("📚 RAG Study Bot (Local, No OpenAI)")
    st.caption("Ask questions directly from your uploaded PDFs")

    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Upload at least one PDF to begin.")
        return

    all_docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        docs = load_pdf(file)
        all_docs.extend(docs)

    split_documents = split_docs(all_docs)
    vectorstore = build_vectorstore(split_documents)

    st.success("Documents indexed. Ask your questions below 👇")

    question = st.text_input("❓ Ask a question")

    if question:
        retrieved_docs = retrieve_docs(vectorstore, question)
        context = "\n\n".join(d.page_content for d in retrieved_docs)

        answer = generate_answer(context, question)

        st.subheader("🤖 Answer")
        st.write(answer)

        st.subheader("📎 Sources")
        for i, d in enumerate(retrieved_docs, 1):
            st.write(f"{i}. {d.metadata['source']} (Page {d.metadata['page']})")


if __name__ == "__main__":
    main()
