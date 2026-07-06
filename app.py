"""Streamlit chat UI for the PDF RAG chatbot.

Run:  streamlit run app.py
"""
import streamlit as st
from src import vector_store
from src.rag import answer

st.set_page_config(page_title="PDF RAG Chatbot", page_icon="📄")
st.title("📄 PDF RAG Chatbot")


@st.cache_resource
def _load_index():
    """Load the FAISS index once per session."""
    return vector_store.load()


try:
    index, chunks = _load_index()
except FileNotFoundError:
    st.warning("No index found. Add PDFs to `data/pdfs/`, then run `python -m src.ingest`.")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Replay past turns
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

question = st.chat_input("Ask a question about your PDFs")
if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching and answering..."):
            result = answer(question, index, chunks)
        st.markdown(result["answer"])
        if result["sources"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.write(f"- {s['source']} — p.{s['page']} (score {s['score']:.2f})")

    st.session_state.history.append({"role": "assistant", "content": result["answer"]})
