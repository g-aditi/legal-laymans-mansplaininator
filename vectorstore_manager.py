import os
from langchain_community.vectorstores import FAISS
from models import embeddings

BASE_DIR = "data"

def _faiss_path(mode: str) -> str:
    return os.path.join(BASE_DIR, f"faiss_{mode}")

def save_faiss(vectorstore, mode: str):
    path = _faiss_path(mode)
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)

def load_faiss(mode: str):
    path = _faiss_path(mode)
    index_file = os.path.join(path, "index.faiss")
    store_file = os.path.join(path, "index.pkl")
    if not (os.path.exists(index_file) and os.path.exists(store_file)):
        return None
    return FAISS.load_local(
        path,
        embeddings,
        allow_dangerous_deserialization=True
    )
