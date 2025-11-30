from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

OLLAMA_URL = "http://127.0.0.1:11434"
LLM_MODEL = "llama3.1"
EMBED_MODEL = "nomic-embed-text"


llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.2
)

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL
)


def build_prompt(question, context):
    return [
        SystemMessage(content="You are an expert RAG assistant. Use ONLY the provided context. If unsure, say 'I don't know'."),
        HumanMessage(content=f"""
CONTEXT:
{context}

QUESTION:
{question}
""")
    ]
