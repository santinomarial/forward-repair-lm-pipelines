import argparse
import json
from collections import defaultdict

from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    CORPUS_PATH,
    EXAMPLES_PATH,
    EXPERIMENT_DEFAULT_SUFFIX,
    OUTPUT_DIR,
    TOP_K,
    experiment_paths,
)
from data_loader import load_jsonl
from llm_backends import build_llm_backend
from retriever import DenseRetriever, build_retriever
from pipeline import ForwardRepairPipeline
from metrics import (
    exact_match,
    contains_answer,
    recall_at_k,
    all_support_recall_at_k,
    recovery_rate,
)


console = Console()


def score_run(run: dict, gold: str, support_doc_ids: list[str]) -> dict:
    return {
        "exact_match": exact_match(run["answer"], gold),
        "contains_answer": contains_answer(run["answer"], gold),
        "recall_at_k": recall_at_k(run["docs"], support_doc_ids),
        "all_support_recall_at_k": all_support_recall_at_k(
            run["docs"],
            support_doc_ids,
        ),
    }


def doc_ids(docs: list[dict]) -> list[str]:
    return [doc["id"] for doc in docs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-suffix", default=EXPERIMENT_DEFAULT_SUFFIX)
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--corrupt-stage", default="query", choices=["query", "answer"])
    parser.add_argument("--retriever", choices=["bm25", "dense"], default="bm25")
    parser.add_argument("--llm", choices=["openai", "ollama"], default="openai")
    parser.add_argument(
        "--model",
        default=None,
        help="Provider model name (defaults to OPENAI_MODEL or OLLAMA_MODEL).",
    )
    parser.add_argument("--ollama-api-base", default=OLLAMA_API_BASE)
    parser.add_argument(
        "--dense-model",
        default=DenseRetriever.DEFAULT_MODEL,
        help="Sentence-transformers model used by --retriever dense.",
    )
    parser.add_argument("--include-iterative", action="store_true",
                        help="Also run the iterative repair condition (query corruption only)")
    args = parser.parse_args()

    results_path, summary_path = experiment_paths(args.output_suffix)

    model = args.model or (OPENAI_MODEL if args.llm == "openai" else OLLAMA_MODEL)
    llm_backend = build_llm_backend(
        args.llm,
        model,
        openai_api_key=OPENAI_API_KEY,
        ollama_api_base=args.ollama_api_base,
    )
    lm_stoch = llm_backend.configure(seed=args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)

    corpus = load_jsonl(CORPUS_PATH)
    examples = load_jsonl(EXAMPLES_PATH)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    retriever = build_retriever(
        backend=args.retriever,
        corpus=corpus,
        top_k=TOP_K,
        dense_model=args.dense_model,
    )
    pipeline = ForwardRepairPipeline(retriever=retriever, corrupt_stage=args.corrupt_stage)

    if args.corrupt_stage == "query":
        pipeline.corrupted_query_generator.generate.set_lm(lm_stoch)
    else:
        pipeline.corrupted_answer_generator.generate.set_lm(lm_stoch)

    if results_path.exists():
        results_path.unlink()

    rows = []

    for example in tqdm(examples, desc="Running experiment"):
        question = example["question"]
        gold = example["answer"]
        support_doc_ids = example["support_doc_ids"]

        baseline = pipeline.run_baseline(question)
        corrupted = pipeline.run_corrupted(question)

        if args.corrupt_stage == "query":
            repaired = pipeline.run_repaired(question=question, bad_query=corrupted["query"])
        else:
            repaired = pipeline.run_repaired(question=question, bad_answer=corrupted["answer"])

        row = {
            "id": example["id"],
            "question": question,
            "gold": gold,
            "support_doc_ids": support_doc_ids,
            "baseline": {
                **baseline,
                "doc_ids": doc_ids(baseline["docs"]),
                "metrics": score_run(baseline, gold, support_doc_ids),
            },
            "corrupted": {
                **corrupted,
                "doc_ids": doc_ids(corrupted["docs"]),
                "metrics": score_run(corrupted, gold, support_doc_ids),
            },
            "repaired": {
                **repaired,
                "doc_ids": doc_ids(repaired["docs"]),
                "metrics": score_run(repaired, gold, support_doc_ids),
            },
        }

        if args.include_iterative and args.corrupt_stage == "query":
            rep_iter = pipeline.run_repaired_iterative(
                question=question, bad_query=corrupted["query"]
            )
            row["repaired_iterative"] = {
                **rep_iter,
                "doc_ids": doc_ids(rep_iter["docs"]),
                "metrics": score_run(rep_iter, gold, support_doc_ids),
            }

        rows.append(row)

        with results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    summary = summarize(rows)

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)


def summarize(rows: list[dict]) -> dict:
    totals = defaultdict(lambda: defaultdict(float))
    _all_modes = ["baseline", "corrupted", "repaired", "repaired_iterative"]
    modes = [m for m in _all_modes if m in rows[0]]

    for row in rows:
        for mode in modes:
            for metric_name, value in row[mode]["metrics"].items():
                totals[mode][metric_name] += value

    n = len(rows)

    summary = {
        mode: {
            metric_name: value / n
            for metric_name, value in totals[mode].items()
        }
        for mode in modes
    }

    summary["recovery"] = recovery_rate(rows)

    return summary


def print_summary(summary: dict) -> None:
    table = Table(title="Forward Repair Results")

    table.add_column("Condition")
    table.add_column("Exact Match")
    table.add_column("Contains Answer")
    table.add_column("Recall@K")
    table.add_column("All Support Recall@K")

    _all_modes = ["baseline", "corrupted", "repaired", "repaired_iterative"]
    for mode in [m for m in _all_modes if m in summary]:
        table.add_row(
            mode,
            f"{summary[mode]['exact_match']:.2f}",
            f"{summary[mode]['contains_answer']:.2f}",
            f"{summary[mode]['recall_at_k']:.2f}",
            f"{summary[mode]['all_support_recall_at_k']:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    main()
