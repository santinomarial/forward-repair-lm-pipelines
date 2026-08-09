import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

EXAMPLES_PATH = DATA_DIR / "hotpot_examples.jsonl"
CORPUS_PATH = DATA_DIR / "hotpot_corpus.jsonl"

EXPERIMENT_DEFAULT_SUFFIX = "hotpot_50"
ANALYSIS_DEFAULT_SUFFIX = "hotpot_50_final"


def experiment_paths(output_suffix: str) -> tuple[Path, Path]:
    """Return result and summary paths for an experiment suffix."""
    return (
        OUTPUT_DIR / f"{output_suffix}_results.jsonl",
        OUTPUT_DIR / f"{output_suffix}_summary.json",
    )


RESULTS_PATH, SUMMARY_PATH = experiment_paths(ANALYSIS_DEFAULT_SUFFIX)
RESCORED_RESULTS_PATH = OUTPUT_DIR / f"{ANALYSIS_DEFAULT_SUFFIX}_results_renormalized.jsonl"

AGGREGATE_SEED_PATHS = tuple(
    OUTPUT_DIR / f"hotpot_300_seed{seed}_results.jsonl" for seed in range(3)
)
AGGREGATED_RESULTS_PATH = OUTPUT_DIR / "hotpot_300_aggregated.json"
STRATIFIED_RESULTS_PATH = OUTPUT_DIR / "stratified_analysis.json"
STRATIFIED_MULTI_RESULTS_PATH = OUTPUT_DIR / "stratified_300x3.json"
STAGE_COMPARISON_PATH = OUTPUT_DIR / "stage_comparison.json"
SIGNIFICANCE_PATH = OUTPUT_DIR / "significance.json"
ROUTER_MODEL_PATH = OUTPUT_DIR / "adaptive_router_model.json"
ROUTER_METRICS_PATH = OUTPUT_DIR / "adaptive_router_metrics.json"

FIGURE_QUERY_SEED_PATHS = tuple(
    OUTPUT_DIR / f"hotpot_300_seed{seed}_results_renormalized.jsonl"
    for seed in range(3)
)
FIGURE_ANSWER_PATH = (
    OUTPUT_DIR / "hotpot_300_answer_corruption_seed0_results_renormalized.jsonl"
)
FIGURE_ITERATIVE_PATH = (
    OUTPUT_DIR / "hotpot_300_iterative_seed0_results_renormalized.jsonl"
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_API_BASE = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")

TOP_K = 5
