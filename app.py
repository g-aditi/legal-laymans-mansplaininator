import streamlit as st
import os
import json
import re
import pandas as pd

from rag_pipeline import load_document, build_vectorstore, get_retriever, rag_query
from utils import ensure_dirs
from models import llm, OLLAMA_URL
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.llms import Ollama  # judge model


ensure_dirs()

st.title("LLM: Legal Layman's Mansplaininator")
st.caption("Jargon simplified")

tab_translator, tab_compliance, tab_evaluator = st.tabs([
    "Translator",
    "Compliance Highlighter",
    "Prompt Evaluator"
])

def _index_ui(mode: str, docs_dir_label: str):
    st.header("Index documents")

    source_type = st.radio(
        "Choose data source:",
        ["Webpage URLs", "Upload Files"],
        key=f"src_{mode}"
    )

    if source_type == "Webpage URLs":
        urls_text = st.text_area(
            "Enter URLs (separate by spaces, commas, semicolons, or new lines):",
            key=f"urls_{mode}"
        )

        if st.button("Load and Index URLs", key=f"index_urls_{mode}"):
            import re
            raw_urls = re.split(r"[,\s;]+", urls_text)
            urls = [u.strip() for u in raw_urls if u.strip()]
            if not urls:
                st.warning("Please enter valid URLs.")
                return

            docs = []
            for url in urls:
                try:
                    loaded = load_document(url)
                    docs.extend(loaded)
                except Exception as e:
                    st.error(f"Failed to load {url}: {e}")

            if docs:
                with st.spinner("Indexing..."):
                    build_vectorstore(docs, mode)
                st.success(f"Indexed {len(docs)} document pieces from {len(urls)} URLs.")

    if source_type == "Upload Files":
        uploaded_files = st.file_uploader(
            "Upload one or more PDF or TXT files",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            key=f"files_{mode}"
        )

        if uploaded_files and st.button("Upload and Index Files", key=f"index_files_{mode}"):

            docs = []
            for file in uploaded_files:
                try:
                    save_path = os.path.join("data", docs_dir_label, file.name)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)

                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    loaded = load_document(save_path)
                    docs.extend(loaded)

                except Exception as e:
                    st.error(f"Failed to load {file.name}: {e}")

            if docs:
                with st.spinner("Indexing..."):
                    build_vectorstore(docs, mode)
                st.success(f"Indexed {len(docs)} document pieces.")

def _chat_ui(mode: str):
    retriever = get_retriever(mode)
    if retriever is None:
        st.info("No FAISS index found. Please index documents first.")
        return

    chat_key = f"chat_{mode}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for i, doc in enumerate(msg["sources"]):
                        snippet = doc.page_content[:300].replace("\n", " ")
                        st.markdown(f"**Chunk {i+1}:** {snippet}...")

    user_input = st.chat_input("Ask a question")
    if user_input:
        st.session_state[chat_key].append({"role": "user", "content": user_input, "sources": None})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = rag_query(user_input, retriever, mode)

            st.write(answer)

            if sources:
                with st.expander("Sources used"):
                    for i, doc in enumerate(sources):
                        snippet = doc.page_content[:300].replace("\n", " ")
                        st.markdown(f"**Chunk {i+1}:** {snippet}...")

        st.session_state[chat_key].append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

with tab_translator:
    st.title("Translator")
    _index_ui("translator", "docs_translator")
    st.divider()
    st.header("Chat")
    _chat_ui("translator")

with tab_compliance:
    st.title("Compliance Highlighter")
    _index_ui("compliance", "docs_compliance")
    st.divider()
    st.header("Chat - Compliance Analysis")
    _chat_ui("compliance")

with tab_evaluator:
    st.title("Prompt Evaluator")

    from langchain_core.messages import SystemMessage, HumanMessage
    from models import llm
    from rag_pipeline import rag_query, get_retriever

    TRANSLATOR_EVAL_QUESTIONS = [
        "Explain what RAM slack and file slack are in simple terms and how investigators use them.",
        "Explain what digital evidence acquisition means in a way a non-technical judge could understand.",
        "Explain the steps investigators take to avoid modifying original digital evidence, in plain English."
    ]

    COMPLIANCE_EVAL_QUESTIONS = [
        "Based on the provided testimony, identify potential chain-of-custody issues.",
        "Does the procedure described comply with NIJ and NIST digital evidence handling standards? Explain.",
        "Identify any steps in the acquisition process that could jeopardize admissibility in court."
    ]

    JUDGE_PROMPT_TEMPLATE = """
    You are an impartial, strict evaluator for digital forensics answers.

    You are given:
    1. CONTEXT (retrieved text from RAG)
    2. QUESTION asked to the model
    3. ANSWER generated by the model

    Score the ANSWER on these metrics, from 1 (poor) to 5 (excellent):

    1. Technical Correctness  
    2. Completeness  
    3. Clarity for Legal Professionals  
    4. Faithfulness to Source Context  
       - Compare ANSWER directly to CONTEXT.
       - Penalize invented details, assumptions, or irrelevant info.
       - Reward answers that stay strictly within the provided CONTEXT.

    Return ONLY JSON:
    {
      "technical_correctness": <1-5>,
      "completeness": <1-5>,
      "clarity_for_legal_professionals": <1-5>,
      "faithfulness_to_source_context": <1-5>,
      "comments": "<short explanation>"
    }

    CONTEXT:
    <<<CONTEXT>>>

    QUESTION:
    <<<QUESTION>>>

    ANSWER:
    <<<ANSWER>>>
    """

    st.subheader("Judge Model Settings (LLM-as-a-Judge)")
    judge_model_name = st.text_input("Judge model (Ollama name)", value="mistral")
    judge_base_url = st.text_input("Judge Base URL", value=OLLAMA_URL)

    if st.button("Initialize Judge Model"):
        try:
            st.session_state["judge_llm"] = Ollama(
                model=judge_model_name,
                base_url=judge_base_url,
                temperature=0.0
            )
            st.success(f"Judge model '{judge_model_name}' initialized.")
        except Exception as e:
            st.error(f"Failed to initialize judge model: {e}")

    judge_llm = st.session_state.get("judge_llm", None)

    def extract_json(text):
        text = text.strip()
        try:
            return json.loads(text)
        except:
            pass
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except:
                return None
        return None

    def run_evaluator(prompts_block, questions, mode):
        """
        mode = 'translator' or 'compliance'
        Uses the corresponding FAISS retriever and RAG pipeline.
        """
        prompts = [p.strip() for p in prompts_block.split("---") if p.strip()]
        if not prompts:
            st.warning("Please provide system prompts separated by '---'.")
            return None, None

        retriever = get_retriever(mode)
        if retriever is None:
            st.error(f"No FAISS index found for mode '{mode}'. Please index documents first.")
            return None, None

        rows = []

        for pid, system_prompt in enumerate(prompts, start=1):
            st.markdown(f"### Prompt #{pid}")
            st.code(system_prompt)

            for q in questions:
                st.write(f"**Q:** {q}")
                answer, sources = rag_query(q, retriever, mode)

                st.markdown("**Generated Answer:**")
                st.write(answer)

                context_text = "\n\n".join([d.page_content for d in sources])

                judge_prompt = (
                    JUDGE_PROMPT_TEMPLATE
                    .replace("<<<CONTEXT>>>", context_text)
                    .replace("<<<QUESTION>>>", q)
                    .replace("<<<ANSWER>>>", answer)
                )

                judge_raw = judge_llm.invoke([SystemMessage(content=judge_prompt)])
                parsed = extract_json(judge_raw)

                if parsed is None:
                    parsed = {
                        "technical_correctness": 1,
                        "completeness": 1,
                        "clarity_for_legal_professionals": 1,
                        "faithfulness_to_source_context": 1,
                        "comments": "Judge output could not be parsed."
                    }

                rows.append({
                    "prompt_id": pid,
                    "prompt_text": system_prompt,
                    "question": q,
                    "answer": answer,
                    "technical_correctness": parsed["technical_correctness"],
                    "completeness": parsed["completeness"],
                    "clarity_for_legal_professionals": parsed["clarity_for_legal_professionals"],
                    "faithfulness_to_source_context": parsed["faithfulness_to_source_context"],
                    "comments": parsed["comments"]
                })

        df = pd.DataFrame(rows)

        summary = (
            df.groupby(["prompt_id", "prompt_text"])[
                ["technical_correctness", "completeness",
                 "clarity_for_legal_professionals", "faithfulness_to_source_context"]
            ].mean().reset_index()
        )

        summary["overall_score"] = summary[
            ["technical_correctness", "completeness",
             "clarity_for_legal_professionals", "faithfulness_to_source_context"]
        ].mean(axis=1)

        return df, summary

    st.header("Translator Prompt Evaluator")
    translator_prompts = st.text_area(
        "Enter translator system prompts (use --- between prompts)",
        height=200
    )

    if st.button("Run Translator Evaluation"):
        if judge_llm is None:
            st.error("Judge model not initialized.")
        else:
            df, summary = run_evaluator(translator_prompts, TRANSLATOR_EVAL_QUESTIONS, "translator")
            if df is not None:
                st.subheader("Raw Results")
                st.dataframe(df)

                st.subheader("Summary")
                st.dataframe(summary.sort_values("overall_score", ascending=False))

    st.header("Compliance Prompt Evaluator")
    compliance_prompts = st.text_area(
        "Enter compliance system prompts (use --- between prompts)",
        height=200
    )

    if st.button("Run Compliance Evaluation"):
        if judge_llm is None:
            st.error("Judge model not initialized.")
        else:
            df, summary = run_evaluator(compliance_prompts, COMPLIANCE_EVAL_QUESTIONS, "compliance")
            if df is not None:
                st.subheader("Raw Results")
                st.dataframe(df)

                st.subheader("Summary")
                st.dataframe(summary.sort_values("overall_score", ascending=False))
