import os
import streamlit as st
from rag.loader import load_and_chunk
from rag.vectorstore import create_vectorstore, load_vectorstore
from rag.chain import build_chain
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="RAG Research Assistant", page_icon="🔍")
st.title("🔍 RAG Research Assistant")
st.caption("Upload a research paper and ask questions about it")

# ── Step 1: Upload PDF ──────────────────────────────
uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

if uploaded_file:
    # Save PDF to disk
    pdf_path = f"data/papers/{uploaded_file.name}"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Index the PDF
    with st.spinner("Reading and indexing PDF..."):
        chunks = load_and_chunk(pdf_path)
        vectorstore = create_vectorstore(chunks)
        st.success(f"✅ Indexed {len(chunks)} chunks from {uploaded_file.name}")

    # ── Step 2: Ask Questions ───────────────────────
    st.markdown("---")
    question = st.text_input("Ask a question about the paper:")

    if question:
        with st.spinner("Searching..."):
            chain = build_chain(vectorstore)
            result = chain.invoke({"query": question})

        # Show answer
        st.markdown("### 📝 Answer")
        st.write(result["result"])

        # Show source chunks
        st.markdown("### 📄 Source Chunks Used")
        for i, doc in enumerate(result["source_documents"]):
            with st.expander(f"Chunk {i+1} — Page {doc.metadata.get('page', '?')}"):
                st.write(doc.page_content)