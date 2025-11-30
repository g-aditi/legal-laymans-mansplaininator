import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    UnstructuredURLLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.vectorstores import FAISS
from models import llm, embeddings, build_prompt
from vectorstore_manager import save_faiss, load_faiss


def load_document(path_or_url):
    """Load a webpage, a PDF (remote or local), or a text file."""

    # ----------------------------------------
    # CASE 1: PDF URL (THIS FIXES YOUR ISSUE)
    # ----------------------------------------
    if path_or_url.lower().endswith(".pdf"):
        loader = PyPDFLoader(path_or_url)
        docs = loader.load()

        if not docs:
            raise ValueError("❌ PDF loader returned no content.")
        return docs

    # ----------------------------------------
    # CASE 2: Regular webpage URL
    # ----------------------------------------
    if path_or_url.startswith("http"):
        loader = UnstructuredURLLoader(urls=[path_or_url])
        docs = loader.load()

        if not docs:
            raise ValueError("❌ Webpage loader returned no content.")
        return docs

    # ----------------------------------------
    # CASE 3: Local file path
    # ----------------------------------------
    if os.path.exists(path_or_url):
        if path_or_url.lower().endswith(".pdf"):
            loader = PyPDFLoader(path_or_url)
        else:
            loader = TextLoader(path_or_url)

        docs = loader.load()

        if not docs:
            raise ValueError("❌ Local file loader returned no content.")
        return docs

    raise ValueError("❌ Invalid input: Not a PDF URL, webpage URL, or existing file.")


def build_vectorstore(docs):
    """Split documents, generate embeddings, and save FAISS index."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    splits = splitter.split_documents(docs)

    if not splits:
        raise ValueError("❌ Text splitter created zero chunks.")

    # Optional early validation: embed first chunk
    test_embed = embeddings.embed_documents([splits[0].page_content])
    if not test_embed:
        raise ValueError("❌ Embedding model returned no embeddings.")

    vectorstore = FAISS.from_documents(splits, embeddings)
    save_faiss(vectorstore)
    return vectorstore


def get_retriever():
    """Load FAISS index (if exists) and return a retriever."""

    vectorstore = load_faiss()
    if vectorstore is None:
        return None

    return vectorstore.as_retriever(search_kwargs={"k": 4})


def rag_query(question, retriever):
    """Retrieve relevant text chunks and generate LLM answer."""

    docs = retriever.invoke(question)

    if isinstance(docs, dict):  
        docs = docs.get("documents", [])

    if not isinstance(docs, list):
        docs = [docs]

    if not docs:
        return "No relevant text found in document.", []

    context = "\n\n".join([d.page_content for d in docs])
    prompt = build_prompt(question, context)

    response = llm.invoke(prompt)
    return response.content, docs
