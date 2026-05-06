import argparse
import json
from collections import defaultdict

import dspy
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
    CORPUS_PATH,
    EXAMPLES_PATH,
    OUTPUT_DIR,
    TOP_K,
)
from data_loader import load_jsonl
from retriever import BM25Retriever
from pipeline import ForwardRepairPipeline
from metrics import (
    exact_match,
    contains_answer,
    recall_at_k,
    all_support_recall_at_k,
)


console = Console()


def configure_dspy(seed: int) -> dspy.LM:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")

    lm_det = dspy.LM(
        model=f"openai/{OPENAI_MODEL}",
        api_key=OPENAI_API_KEY,
        temperature=0,
        max_tokens=300,
    )
    dspy.configure(lm=lm_det)

    lm_stoch = dspy.LM(
        model=f"openai/{OPENAI_MODEL}",
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=300,
        seed=seed,
    )
    return lm_stoch


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
    parser.add_argument("--output-suffix", default="hotpot_50")
    parser.add_argument("--max-examples", type=int, default=None)
    parser.add_argument("--corrupt-stage", default="query", choices=["query", "answer"])
    parser.add_argument("--include-iterative", action="store_true",
                        help="Also run the iterative repair condition (query corruption only)")
    args = parser.parse_args()

    results_path = OUTPUT_DIR / f"{args.output_suffix}_results.jsonl"
    summary_path = OUTPUT_DIR / f"{args.output_suffix}_summary.json"

    lm_stoch = configure_dspy(seed=args.seed)
    OUTPUT_DIR.mkdir(exist_ok=True)

    corpus = load_jsonl(CORPUS_PATH)
    examples = load_jsonl(EXAMPLES_PATH)
    if args.max_examples is not None:
        examples = examples[: args.max_examples]

    retriever = BM25Retriever(corpus=corpus, top_k=TOP_K)
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

    corrupted_broken = 0
    repaired_fixed = 0

    for row in rows:
        corrupted_correct = row["corrupted"]["metrics"]["exact_match"] == 1
        repaired_correct = row["repaired"]["metrics"]["exact_match"] == 1

        if not corrupted_correct:
            corrupted_broken += 1
            if repaired_correct:
                repaired_fixed += 1

    summary["recovery"] = {
        "corrupted_broken_count": corrupted_broken,
        "repaired_fixed_count": repaired_fixed,
        "recovery_rate": repaired_fixed / corrupted_broken if corrupted_broken else 0.0,
    }

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