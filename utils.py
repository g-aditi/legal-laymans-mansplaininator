import os

def ensure_dirs():
    os.makedirs("data/docs", exist_ok=True)
    os.makedirs("data/faiss_index", exist_ok=True)
