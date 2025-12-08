import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredURLLoader,
    TextLoader
)

from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document


from models import (
    llm,
    embeddings,
    build_translator_prompt,
    build_compliance_prompt
)

from vectorstore_manager import save_faiss, load_faiss

def clean_text(text: str) -> str:
    """Normalize text extracted from PDFs to improve embeddings."""
    if not text:
        return ""

    text = re.sub(r"Page\s+\d+|\d+\s+Page", "", text, flags=re.IGNORECASE)
    text = re.sub(r"U\.?S\.? Department of Justice.*", "", text)
    text = re.sub(r"NIST.*Interagency.*Report.*", "", text)

    text = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", text)

    text = text.replace("\n", " ")

    text = re.sub(r"\s{2,}", " ", text)

    return text.strip()

def load_pdf_with_fallback(path: str):
    """
    Load PDFs using PyMuPDFLoader (best for DOJ/NIST PDF structure).
    No OCR fallback is used because unstructured OCR is not installed.
    """

    print("\n=== Loading PDF with PyMuPDFLoader ===")
    loader = PyMuPDFLoader(path)
    docs = loader.load()

    combined_len = sum(len(d.page_content.strip()) for d in docs)
    print(f"PyMuPDF extracted {combined_len} characters from {len(docs)} pages.")

    if combined_len < 200:
        print("Warning: Extracted very little text. PDF may be image-based or poorly structured.")
        print("Proceeding anyway with available text.")

    return docs

def load_document(path_or_url):
    """Load URL, PDF, or text file with normalization."""

    if path_or_url.lower().endswith(".pdf"):
        docs = load_pdf_with_fallback(path_or_url)

        cleaned_docs = []
        for d in docs:
            cleaned_text = clean_text(d.page_content)
            cleaned_docs.append(Document(page_content=cleaned_text, metadata=d.metadata))

        print("\n[LOG] Sample cleaned text:")
        print(cleaned_docs[0].page_content[:600])
        print("-------------------------------------------------------")

        return cleaned_docs

    if path_or_url.startswith("http"):
        loader = UnstructuredURLLoader(urls=[path_or_url])
        docs = loader.load()

        cleaned_docs = [
            Document(page_content=clean_text(d.page_content), metadata=d.metadata)
            for d in docs
        ]
        return cleaned_docs

    if os.path.exists(path_or_url):
        loader = TextLoader(path_or_url)
        docs = loader.load()

        cleaned_docs = [
            Document(page_content=clean_text(d.page_content), metadata=d.metadata)
            for d in docs
        ]
        return cleaned_docs

    raise ValueError("Invalid input: not a valid PDF, URL, or existing file.")


def build_vectorstore(docs, mode: str):
    """Split normalized documents, build embeddings, save FAISS index."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )

    splits = splitter.split_documents(docs)

    if not splits:
        raise ValueError("Text splitter created zero chunks. PDF extraction may have failed.")

    print(f"[LOG] Created {len(splits)} chunks.")

    test_embed = embeddings.embed_documents([splits[0].page_content])
    if not test_embed:
        raise ValueError("Embedding model returned no embeddings.")

    vectorstore = FAISS.from_documents(splits, embeddings)
    save_faiss(vectorstore, mode)

    return vectorstore


def get_retriever(mode: str):
    vectorstore = load_faiss(mode)
    if vectorstore is None:
        return None
    return vectorstore.as_retriever(search_kwargs={"k": 4})


def rag_query(question: str, retriever, mode: str):
    """Retrieve relevant text chunks and generate an answer (no confidence returned)."""

    if retriever is None:
        return "No FAISS index loaded for this mode.", []

    try:
        results = retriever.vectorstore.similarity_search_with_score(question, k=4)
        docs = [doc for doc, dist in results]
    except Exception:
        docs = retriever.get_relevant_documents(question)

    if not docs:
        return "No relevant text found in document.", []

    context = "\n\n".join([d.page_content for d in docs])

    if mode == "translator":
        prompt = build_translator_prompt(question, context)
    else:
        prompt = build_compliance_prompt(question, context)

    answer = llm.invoke(prompt)

    return answer, docs
