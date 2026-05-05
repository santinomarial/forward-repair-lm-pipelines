from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

EXAMPLES_PATH = DATA_DIR / "hotpot_examples.jsonl"
CORPUS_PATH = DATA_DIR / "hotpot_corpus.jsonl"

RESULTS_PATH = OUTPUT_DIR / "results.jsonl"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

TOP_K = 5