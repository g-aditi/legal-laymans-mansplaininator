import socket
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage

OLLAMA_URL = f"http://{socket.gethostname()}:11434"
LLM_MODEL = "llama3.2"
EMBED_MODEL = "mxbai-embed-large"

llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_URL,
    temperature=0.2
)

embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_URL
)

def build_translator_prompt(question, context):
    """
    Prompt for translating cybercrime / evidence collection jargon into layman's terms.
    """
    system = SystemMessage(content=(
        "You are a cybersecurity translator. Use only the provided context to explain "
        "technical terms and procedures in clear, concise, plain English suitable for a layperson. "
        "When you are not certain, say 'I don't know'. Do not invent facts."
    ))
    human = HumanMessage(content=f"""
CONTEXT:
{context}

INSTRUCTION:
Rewrite the context or answer the question below in simple, everyday English that a non-technical legal layperson can understand. Keep explanations concise and avoid jargon.

QUESTION:
{question}
""")
    return [system, human]

def build_compliance_prompt(question, context):
    """
    Prompt for evaluating admissibility/compliance of a digital evidence collector's testimony.
    """
    system = SystemMessage(content=(
        "You are an expert in digital evidence admissibility and chain-of-custody issues. "
        "Use only the provided context (testimony and supporting text) to analyze compliance with typical legal standards for admissibility. "
        "State whether the testimony appears admissible, list any specific problems, and cite the relevant parts of the provided context. "
        "If unsure, say 'I don't know' and list the missing information needed to decide."
    ))
    human = HumanMessage(content=f"""
TESTIMONY / CONTEXT:
{context}

INSTRUCTION:
Based only on the context, answer whether the testimony meets typical standards for admissibility (chain of custody, proper procedures, authorization, integrity of evidence, etc.). Give a short conclusion (Admissible / Potential issues / Not admissible) and a bullet list with the specific reasons or missing items.
QUESTION:
{question}
""")
    return [system, human]
