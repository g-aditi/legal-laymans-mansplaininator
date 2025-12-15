import os

def ensure_dirs():
    os.makedirs("data/docs_translator", exist_ok=True)
    os.makedirs("data/docs_compliance", exist_ok=True)
    os.makedirs("data/faiss_translator", exist_ok=True)
    os.makedirs("data/faiss_compliance", exist_ok=True)
