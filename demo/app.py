"""Minimal interactive demonstration of localized forward repair."""

import sys
from pathlib import Path

import dspy
import streamlit as st

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    CORPUS_PATH,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    TOP_K,
)
from data_loader import load_jsonl  # noqa: E402
from llm_backends import build_llm_backend  # noqa: E402
from pipeline import ForwardRepairPipeline  # noqa: E402
from retriever import DenseRetriever, build_retriever  # noqa: E402


st.set_page_config(page_title="Localized Forward Repair", page_icon="🔧", layout="wide")


@st.cache_resource(show_spinner="Building retrieval index…")
def get_retriever(name: str, top_k: int, dense_model: str):
    corpus = load_jsonl(CORPUS_PATH)
    return build_retriever(name, corpus, top_k, dense_model=dense_model)


def configure_pipeline(
    llm_name: str,
    model: str,
    api_key: str | None,
    ollama_api_base: str,
    retriever_name: str,
    dense_model: str,
    top_k: int,
    corrupt_stage: str,
) -> ForwardRepairPipeline:
    backend = build_llm_backend(
        llm_name,
        model,
        openai_api_key=api_key,
        ollama_api_base=ollama_api_base,
    )
    models = backend.create_models(seed=0)
    dspy.configure(lm=models.deterministic)
    pipeline = ForwardRepairPipeline(
        get_retriever(retriever_name, top_k, dense_model),
        corrupt_stage=corrupt_stage,
    )
    if corrupt_stage == "query":
        pipeline.corrupted_query_generator.generate.set_lm(models.stochastic)
    else:
        pipeline.corrupted_answer_generator.generate.set_lm(models.stochastic)
    return pipeline


def show_documents(docs: list[dict]) -> None:
    for rank, doc in enumerate(docs, start=1):
        label = f"{rank}. {doc['title']} · score {doc.get('score', 0):.3f}"
        with st.expander(label):
            st.write(doc["text"])


def show_run(run: dict) -> None:
    st.markdown("**Generated query**")
    st.code(run.get("query") or f"{run.get('query_a')}\n{run.get('query_b')}")
    st.markdown("**Answer**")
    st.success(run["answer"])
    st.markdown("**Retrieved documents**")
    show_documents(run["docs"])


st.title("🔧 Localized Forward Repair")
st.caption(
    "Generate → retrieve → answer, then corrupt one stage and repair only that stage."
)

with st.sidebar:
    st.header("Backends")
    llm_name = st.selectbox("LLM", ["openai", "ollama"])
    default_model = OPENAI_MODEL if llm_name == "openai" else OLLAMA_MODEL
    model = st.text_input("Model", value=default_model)
    api_key_input = ""
    ollama_api_base = OLLAMA_API_BASE
    if llm_name == "openai":
        api_key_input = st.text_input(
            "OpenAI API key",
            type="password",
            placeholder="Uses OPENAI_API_KEY when blank",
        )
    else:
        ollama_api_base = st.text_input("Ollama endpoint", value=OLLAMA_API_BASE)

    retriever_name = st.selectbox("Retriever", ["bm25", "dense"])
    dense_model = DenseRetriever.DEFAULT_MODEL
    if retriever_name == "dense":
        dense_model = st.text_input("Embedding model", value=dense_model)
    top_k = st.slider("Top K", min_value=1, max_value=10, value=TOP_K)
    corrupt_stage = st.radio("Corrupt stage", ["query", "answer"], horizontal=True)

question = st.text_area(
    "HotpotQA-style question",
    value="Are Giuseppe Verdi and Ambroise Thomas both opera composers?",
    height=90,
)

configuration = (
    llm_name,
    model,
    retriever_name,
    dense_model,
    top_k,
    corrupt_stage,
    ollama_api_base,
)

if st.button("Run baseline", type="primary", use_container_width=True):
    try:
        with st.spinner("Running baseline pipeline…"):
            pipeline = configure_pipeline(
                llm_name,
                model,
                api_key_input or OPENAI_API_KEY,
                ollama_api_base,
                retriever_name,
                dense_model,
                top_k,
                corrupt_stage,
            )
            st.session_state.pipeline = pipeline
            st.session_state.configuration = configuration
            st.session_state.baseline = pipeline.run_baseline(question)
            st.session_state.question = question
            st.session_state.pop("corrupted", None)
            st.session_state.pop("repaired", None)
    except Exception as exc:
        st.error(f"Baseline failed: {exc}")

pipeline_ready = (
    "pipeline" in st.session_state
    and st.session_state.get("configuration") == configuration
    and st.session_state.get("question") == question
)

if "baseline" in st.session_state and pipeline_ready:
    st.subheader("Baseline")
    show_run(st.session_state.baseline)

if st.button(
    "Trigger corruption + localized repair",
    disabled=not pipeline_ready,
    use_container_width=True,
):
    try:
        with st.spinner("Corrupting and repairing…"):
            pipeline = st.session_state.pipeline
            corrupted = pipeline.run_corrupted(question)
            if corrupt_stage == "query":
                repaired = pipeline.run_repaired(question, bad_query=corrupted["query"])
            else:
                repaired = pipeline.run_repaired(question, bad_answer=corrupted["answer"])
            st.session_state.corrupted = corrupted
            st.session_state.repaired = repaired
    except Exception as exc:
        st.error(f"Corruption/repair failed: {exc}")

if "corrupted" in st.session_state and "repaired" in st.session_state and pipeline_ready:
    st.subheader("Before and after repair")
    before, after = st.columns(2)
    with before:
        st.markdown("### Corrupted")
        show_run(st.session_state.corrupted)
    with after:
        st.markdown("### Repaired")
        show_run(st.session_state.repaired)

if not pipeline_ready and "baseline" in st.session_state:
    st.info("Configuration or question changed. Run the baseline again before repair.")
