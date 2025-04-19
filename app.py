import streamlit as st
from rag_utils import load_documents, setup_vectorstore, get_qa_chain
import os

st.set_page_config(page_title="Aspire RAG QA", layout="centered")

st.title("📘 Aspire RAG QA App")
st.write("Ask questions based on local tutoring documents.")

# Load docs once
if not os.path.exists("chroma_db"):
    with st.spinner("Setting up vector store..."):
        docs = load_documents()
        setup_vectorstore(docs)

# Question input
query = st.text_input("Enter your question:", placeholder="e.g. What are the class timings?")
if query:
    with st.spinner("Thinking..."):
        qa_chain = get_qa_chain()
        answer = qa_chain.run(query)
        st.success(answer)
