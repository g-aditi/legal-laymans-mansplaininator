import streamlit as st
import os
from rag_pipeline import load_document, build_vectorstore, get_retriever, rag_query
from utils import ensure_dirs

ensure_dirs()

st.title("🔥 Local RAG Chat (Llama3.1 + FAISS + Streamlit)")
st.caption("Load webpages, PDFs, or text files and chat with them locally.")

# Sidebar: Load data
st.sidebar.header("📄 Load Data")
source_type = st.sidebar.radio("Choose data source:", ["Webpage URL", "Upload File"])

# --- Load Webpage ---
if source_type == "Webpage URL":
    url = st.sidebar.text_input("Enter URL")

    if st.sidebar.button("Load Webpage"):
        try:
            with st.spinner("Loading webpage..."):
                docs = load_document(url)
                build_vectorstore(docs)
            st.sidebar.success("Indexed webpage successfully!")
        except Exception as e:
            st.sidebar.error(str(e))

# --- Upload File ---
if source_type == "Upload File":
    uploaded = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded and st.sidebar.button("Process File"):
        try:
            path = os.path.join("data/docs", uploaded.name)
            with open(path, "wb") as f:
                f.write(uploaded.getbuffer())

            with st.spinner("Indexing file..."):
                docs = load_document(path)
                build_vectorstore(docs)

            st.sidebar.success("Indexed file successfully!")
        except Exception as e:
            st.sidebar.error(str(e))

# --- Chat Interface ---
st.subheader("💬 Chat With Your Data")

retriever = get_retriever()

if retriever is None:
    st.info("⚠️ No FAISS index found. Load a webpage or upload a file first.")
else:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ask a question")

    if st.button("Send"):
        answer, sources = rag_query(user_input, retriever)
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Assistant", answer))

        st.subheader("📚 Sources Used")
        for i, doc in enumerate(sources):
            st.markdown(f"**Chunk {i+1}:**\n{doc.page_content[:300]}...")

    for speaker, msg in st.session_state.chat_history:
        st.markdown(f"**{speaker}:** {msg}")
