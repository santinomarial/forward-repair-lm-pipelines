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
from telemetry import LMUsageSnapshot
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
    lm_models = llm_backend.create_models(seed=args.seed)
    import dspy

    dspy.configure(lm=lm_models.deterministic)
    lm_stoch = lm_models.stochastic
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

        usage = LMUsageSnapshot([lm_models.deterministic, lm_models.stochastic])
        baseline = pipeline.run_baseline(question)
        baseline["telemetry"].update(usage.finish())

        usage = LMUsageSnapshot([lm_models.deterministic, lm_models.stochastic])
        corrupted = pipeline.run_corrupted(question)
        corrupted["telemetry"].update(usage.finish())

        usage = LMUsageSnapshot([lm_models.deterministic, lm_models.stochastic])
        if args.corrupt_stage == "query":
            repaired = pipeline.run_repaired(question=question, bad_query=corrupted["query"])
        else:
            repaired = pipeline.run_repaired(question=question, bad_answer=corrupted["answer"])
        repaired["telemetry"].update(usage.finish())

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
            usage = LMUsageSnapshot([lm_models.deterministic, lm_models.stochastic])
            rep_iter = pipeline.run_repaired_iterative(
                question=question, bad_query=corrupted["query"]
            )
            rep_iter["telemetry"].update(usage.finish())
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
    summary["instrumentation"] = summarize_instrumentation(rows, modes)

    return summary


def summarize_instrumentation(rows: list[dict], modes: list[str]) -> dict:
    result = {
        "cost_note": (
            "estimated_cost_usd is DSPy/LiteLLM provider-reported cost; "
            "cache hits and local calls may report zero"
        )
    }
    for mode in modes:
        telemetry = [row[mode]["telemetry"] for row in rows]
        n = len(telemetry)
        latency_keys = ["query_generation", "retrieval", "answer_generation"]
        total_calls = sum(int(item["llm_calls"]) for item in telemetry)
        total_tokens = sum(int(item["total_tokens"]) for item in telemetry)
        total_cost = sum(float(item["estimated_cost_usd"]) for item in telemetry)
        total_wall = sum(float(item["wall_clock_seconds"]) for item in telemetry)
        result[mode] = {
            "examples": n,
            "llm_calls": {"total": total_calls, "per_example": total_calls / n},
            "tokens": {
                "prompt_total": sum(int(item["prompt_tokens"]) for item in telemetry),
                "completion_total": sum(
                    int(item["completion_tokens"]) for item in telemetry
                ),
                "total": total_tokens,
                "per_example": total_tokens / n,
            },
            "estimated_cost_usd": {
                "total": total_cost,
                "per_example": total_cost / n,
            },
            "wall_clock_seconds": {
                "total": total_wall,
                "per_example": total_wall / n,
            },
            "stage_latency_seconds": {
                key: {
                    "total": sum(
                        float(item["latency_seconds"][key]) for item in telemetry
                    ),
                    "per_example": sum(
                        float(item["latency_seconds"][key]) for item in telemetry
                    )
                    / n,
                }
                for key in latency_keys
            },
        }
    return result


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

    instrumentation = summary.get("instrumentation", {})
    if instrumentation:
        usage_table = Table(title="Cost and Latency per Condition")
        usage_table.add_column("Condition")
        usage_table.add_column("Calls/example")
        usage_table.add_column("Tokens/example")
        usage_table.add_column("Cost/example")
        usage_table.add_column("Latency/example")
        for mode in [m for m in _all_modes if m in instrumentation]:
            item = instrumentation[mode]
            usage_table.add_row(
                mode,
                f"{item['llm_calls']['per_example']:.1f}",
                f"{item['tokens']['per_example']:.0f}",
                f"${item['estimated_cost_usd']['per_example']:.6f}",
                f"{item['wall_clock_seconds']['per_example']:.3f}s",
            )
        console.print(usage_table)


if __name__ == "__main__":
    main()
