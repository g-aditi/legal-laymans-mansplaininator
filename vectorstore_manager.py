import os
from langchain_community.vectorstores import FAISS
from models import embeddings

FAISS_DIR = "data/faiss_index"


def save_faiss(vectorstore):
    """Save FAISS index to disk."""
    os.makedirs(FAISS_DIR, exist_ok=True)
    vectorstore.save_local(FAISS_DIR)


def load_faiss():
    """Load FAISS index only if files exist. Otherwise return None."""
    index_file = os.path.join(FAISS_DIR, "index.faiss")
    store_file = os.path.join(FAISS_DIR, "index.pkl")

    if not (os.path.exists(index_file) and os.path.exists(store_file)):
        return None

    return FAISS.load_local(
        FAISS_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )
