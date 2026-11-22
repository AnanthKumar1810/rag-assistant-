"""Streamlit interface for the local PDF research assistant."""
import hashlib
import os

import streamlit as st
from dotenv import load_dotenv

from rag.local_llm import LocalOllama
from rag.native import create_vectorstore, load_and_chunk
from rag.structured import ask_structured

load_dotenv()
st.set_page_config(page_title="RAG Research Assistant")
st.title("RAG Research Assistant")
st.caption("Upload a research paper and ask questions about its contents")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")
if uploaded_file:
    file_bytes = uploaded_file.getvalue()
    file_id = hashlib.sha256(file_bytes).hexdigest()
    if st.session_state.get("indexed_pdf") != file_id:
        os.makedirs("data/papers", exist_ok=True)
        pdf_path = f"data/papers/{file_id}.pdf"
        with open(pdf_path, "wb") as output:
            output.write(file_bytes)
        with st.spinner("Reading and indexing PDF..."):
            chunks = load_and_chunk(pdf_path)
            st.session_state["vectorstore"] = create_vectorstore(chunks)
            st.session_state["chunk_count"] = len(chunks)
            st.session_state["indexed_pdf"] = file_id

    st.success(f"Indexed {st.session_state['chunk_count']} passages from {uploaded_file.name}")
    question = st.text_input("Ask a question about the paper")
    if question:
        if os.getenv("ANTHROPIC_API_KEY"):
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(model="claude-sonnet-4-6", anthropic_api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            llm = LocalOllama()
        with st.spinner("Finding evidence and checking the answer..."):
            try:
                answer, sources = ask_structured(st.session_state["vectorstore"], question, llm)
            except (ValueError, OSError, KeyError) as exc:
                st.error(f"Could not verify an answer: {exc}")
                st.stop()
        st.subheader("Answer")
        st.write(answer.answer if answer.found else "Not found in document.")
        if answer.found:
            st.caption("Evidence quotes: " + "; ".join(answer.evidence_quotes))
        st.subheader("Retrieved source passages")
        for number, doc in enumerate(sources, 1):
            with st.expander(f"Passage {number} - PDF page {doc.metadata['page'] + 1}"):
                st.write(doc.page_content)

