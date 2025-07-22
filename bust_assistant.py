# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 18:27:10 2025

@author: loalus01
"""
import streamlit as st
import fitz  # PyMuPDF
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import os

# Create a folder to store uploaded docs
UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Load pre-existing documents and embed them
@st.cache_resource
def load_index():
    docs_path = "docs"
    chunks = []
    for filename in os.listdir(docs_path):
        if filename.endswith(".pdf"):
            doc = fitz.open(os.path.join(docs_path, filename))
            text = "\n".join(page.get_text() for page in doc)
            words = text.split()
            for i in range(0, len(words), 400):
                chunk = " ".join(words[i:i+500])
                chunks.append(chunk)

    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, index, chunks

# Embed and index uploaded PDFs
def process_uploaded_pdfs(uploaded_files, embedder, save_to_disk=False):
    all_chunks = []
    for uploaded_file in uploaded_files:
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            continue

        text = "\n".join(page.get_text() for page in doc)
        words = text.split()
        chunks = [" ".join(words[i:i+500]) for i in range(0, len(words), 400)]
        all_chunks.extend(chunks)

        if save_to_disk:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

    if not all_chunks:
        st.warning("No valid content extracted from uploaded PDFs.")
        return None, []

    embeddings = embedder.encode(all_chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(np.array(embeddings).shape[1])
    index.add(np.array(embeddings))
    return index, all_chunks

def retrieve_context(query, model, index, chunks, k=5):
    q_embed = model.encode([query])
    D, I = index.search(np.array(q_embed), k)
    return "\n\n".join([chunks[i] for i in I[0]])

def construct_prompt(query, context):
    return f"""As a Business Education teacher. Use the context below to answer the user's question.

Context:
{context}

Question: {query}
Answer:"""

def query_ollama(prompt):
    payload = {
        "model": "llama3.2:latest",
        "prompt": prompt,
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    return response.json().get("response", "Error: No response from Ollama")

# Streamlit UI
st.title("📘 Business Education AI Assistant")
st.markdown("Ask a question, or upload PDFs to analyze and query from.")

# State management for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

uploaded_files = st.file_uploader("📂 Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    st.session_state.uploaded_files = uploaded_files

# Clear button
if st.button("🗑️ Clear Uploaded Files"):
    st.session_state.uploaded_files = []
    st.experimental_rerun()

embedder, base_index, base_chunks = load_index()
user_query = st.text_input("💬 Enter your question:")

if user_query:
    with st.spinner("🔎 Retrieving context and querying model..."):
        if st.session_state.uploaded_files:
            result = process_uploaded_pdfs(st.session_state.uploaded_files, embedder, save_to_disk=True)
            if result is None or result[0] is None:
                st.stop()
            user_index, user_chunks = result
            context = retrieve_context(user_query, embedder, user_index, user_chunks)
        else:
            context = retrieve_context(user_query, embedder, base_index, base_chunks)

        prompt = construct_prompt(user_query, context)
        answer = query_ollama(prompt)

    st.subheader("📖 Answer")
    st.write(answer)

    with st.expander("🧠 Retrieved Context"):
        st.write(context)



 