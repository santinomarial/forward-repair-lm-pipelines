"""Compare answer revision and regeneration on identical saved evidence.

No query generation, retrieval, or new corruption is performed. The primary
contrast is blind_revision minus revision: the revision instructions are held
fixed and only exposure to the previous answer changes. Fresh generation is a
secondary, practical comparison with a different task instruction.
"""

import argparse
import hashlib
import json
import random
import re
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import dspy
import numpy as np
from tqdm import tqdm

from config import (
    ANSWER_ABLATION_DEFAULT_SUFFIX,
    FIGURE_ANSWER_PATH,
    OLLAMA_API_BASE,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    experiment_paths,
)
from data_loader import load_jsonl
from dspy_modules import (
    AnswerGenerator, AnswerRepairer, BlindAnswerRepairer, guarded_answer_context,
)
from llm_backends import build_llm_backend
from metrics import contains_answer, exact_match, normalize
from significance import paired_binary_difference
from telemetry import LMUsageSnapshot, summarize_instrumentation


CONDITIONS = ("revision", "blind_revision", "fresh")
PROTOCOL_VERSION = 1


def answer_metrics(answer: str, gold: str) -> dict:
    return {
        "exact_match": exact_match(answer, gold),
        "contains_answer": contains_answer(answer, gold),
        "abstained": int(normalize(answer) == "unknown"),
    }


def validate_sources(rows: list[dict]) -> None:
    if not rows:
        raise ValueError("input contains no rows")
    seen = set()
    for row in rows:
        row_id = str(row["id"])
        if row_id in seen:
            raise ValueError(f"duplicate question ID: {row_id}")
        seen.add(row_id)
        for field in ("question", "gold"):
            if not isinstance(row[field], str) or not row[field].strip():
                raise ValueError(f"{row_id}: {field} must be nonempty text")
        corrupted = row["corrupted"]
        for field in ("context", "query", "answer"):
            if not isinstance(corrupted[field], str):
                raise ValueError(f"{row_id}: corrupted {field} must be text")
        if not isinstance(corrupted["docs"], list):
            raise ValueError(f"{row_id}: corrupted docs must be a list")


def build_modules() -> dict:
    return {
        "revision": AnswerRepairer(),
        "blind_revision": BlindAnswerRepairer(),
        "fresh": AnswerGenerator(),
    }


def run_example(row: dict, modules: dict, models: list, *, seed: int) -> dict:
    """Call each arm once; gold labels are used only after generation."""
    source = row["corrupted"]
    order = list(CONDITIONS)
    random.Random(f"{seed}:{row['id']}").shuffle(order)
    result = {
        "id": str(row["id"]),
        "question": row["question"],
        "gold": row["gold"],
        "evidence": {key: source[key] for key in ("query", "docs", "context")},
        "corrupted": {
            "answer": source["answer"],
            "metrics": answer_metrics(source["answer"], row["gold"]),
        },
        "condition_order": order,
    }
    for mode in order:
        inputs = {"question": row["question"], "context": source["context"]}
        if mode == "revision":
            inputs["bad_answer"] = source["answer"]
        usage = LMUsageSnapshot(models)
        started_at = perf_counter()
        answer = modules[mode](**inputs).answer
        elapsed = perf_counter() - started_at
        result[mode] = {
            "answer": answer,
            "metrics": answer_metrics(answer, row["gold"]),
            "telemetry": {
                **usage.finish(),
                "wall_clock_seconds": elapsed,
                "latency_seconds": {
                    "query_generation": 0.0,
                    "retrieval": 0.0,
                    "answer_generation": elapsed,
                },
            },
        }
    return result


def summarize_ablation(
    rows: list[dict], *, n_resamples: int = 10_000, confidence: float = 0.95, seed: int = 0
) -> dict:
    if not rows:
        raise ValueError("input contains no rows")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate question IDs invalidate paired analysis")
    # Recompute scores so old normalization or edited cached metrics cannot leak in.
    scores = {
        mode: [answer_metrics(row[mode]["answer"], row["gold"]) for row in rows]
        for mode in ("corrupted", *CONDITIONS)
    }
    em = {mode: [score["exact_match"] for score in values] for mode, values in scores.items()}
    broken = [i for i, value in enumerate(em["corrupted"]) if value == 0]
    correct = [i for i, value in enumerate(em["corrupted"]) if value == 1]
    summary: dict = {
        "examples": len(rows),
        "corrupted_broken_count": len(broken),
        "corrupted_correct_count": len(correct),
        "conditions": {},
        "comparisons": {},
        "statistics": {
            "method": "paired percentile bootstrap by question",
            "confidence": confidence,
            "n_resamples": n_resamples,
            "seed": seed,
            "primary_contrast": "blind_revision_minus_revision",
            "note": "Pointwise intervals; secondary contrasts are exploratory, not multiplicity-adjusted.",
        },
    }
    for mode in ("corrupted", *CONDITIONS):
        recovered = sum(em[mode][i] for i in broken)
        damaged = sum(1 - em[mode][i] for i in correct)
        summary["conditions"][mode] = {
            "exact_match": float(np.mean(em[mode])),
            "contains_answer": float(np.mean([s["contains_answer"] for s in scores[mode]])),
            "abstention_rate": float(np.mean([s["abstained"] for s in scores[mode]])),
            "recovered_count": recovered,
            "recovery_rate": recovered / len(broken) if broken else None,
            "damaged_count": damaged,
            "damage_rate": damaged / len(correct) if correct else None,
        }

    def difference(reference: str, candidate: str, indices: list[int]) -> dict:
        return paired_binary_difference(
            [em[reference][i] for i in indices],
            [em[candidate][i] for i in indices],
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )

    for reference, candidate in (
        ("revision", "blind_revision"),
        ("revision", "fresh"),
        *(("corrupted", mode) for mode in CONDITIONS),
    ):
        summary["comparisons"][f"{candidate}_minus_{reference}"] = {
            "exact_match": difference(reference, candidate, list(range(len(rows)))),
            "recovery": difference(reference, candidate, broken),
        }
    instrumentation = summarize_instrumentation(rows, list(CONDITIONS))
    for mode in CONDITIONS:
        instrumentation[mode]["wall_clock_seconds"]["p95"] = float(
            np.quantile([row[mode]["telemetry"]["wall_clock_seconds"] for row in rows], 0.95)
        )
        instrumentation[mode]["priced_calls"] = sum(
            row[mode]["telemetry"].get("priced_calls", 0) for row in rows
        )
    summary["instrumentation"] = instrumentation
    summary["cost_scope"] = (
        "Incremental answer generation only. Saved query, retrieval, and corruption are excluded. "
        "Unpriced calls are not evidence of free inference."
    )
    if "run_config" in rows[0]:
        config = rows[0]["run_config"]
        if any(row.get("run_config") != config for row in rows):
            raise ValueError("cannot combine different run configurations")
        summary["run_config"] = config
        summary["complete"] = len(rows) == config["examples"]
    return summary


def prepare_output(path: Path, sources: list[dict], config: dict, *, resume: bool) -> list[dict]:
    """Never replace existing runs; resume only an exact, complete-row prefix."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8"):
            pass
        return []
    if not resume:
        raise FileExistsError(f"{path} already exists; use --resume or another --output-suffix")
    try:
        rows = load_jsonl(path)
    except json.JSONDecodeError as exc:
        raise ValueError("incomplete JSONL record; preserve this file and use a new output suffix") from exc
    if len(rows) > len(sources):
        raise ValueError("resume input has fewer examples than the saved run")
    for saved, source in zip(rows, sources):
        if saved["id"] != str(source["id"]) or saved.get("run_config") != config:
            raise ValueError("resume requires the same input, prompts, and run configuration")
        if any(mode not in saved for mode in CONDITIONS):
            raise ValueError("resume contains an incomplete condition set")
    return rows


def write_summary(path: Path, rows: list[dict], args: argparse.Namespace) -> None:
    summary = summarize_ablation(
        rows, n_resamples=args.resamples, confidence=args.confidence, seed=args.seed
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for mode, metrics in summary["conditions"].items():
        print(f"{mode:16s} EM={metrics['exact_match']:.1%}")
    primary = summary["comparisons"]["blind_revision_minus_revision"]["exact_match"]
    print(f"Primary EM difference: {primary['estimate']:+.1%}, CI {primary['ci']}")
    print(f"Summary: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FIGURE_ANSWER_PATH)
    parser.add_argument("--output-suffix", default=ANSWER_ABLATION_DEFAULT_SUFFIX)
    parser.add_argument("--max-examples", type=int, default=10)
    parser.add_argument("--llm", choices=["openai", "ollama"], default="openai")
    parser.add_argument("--model")
    parser.add_argument("--ollama-api-base", default=OLLAMA_API_BASE)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--analyze-only", type=Path, help="Re-score an ablation JSONL without LLM calls.")
    args = parser.parse_args()
    if not re.fullmatch(r"[\w-]+", args.output_suffix, flags=re.ASCII):
        parser.error("--output-suffix must contain only letters, digits, underscores, or hyphens")
    if min(args.max_examples, args.max_tokens, args.resamples) <= 0 or not 0 < args.confidence < 1:
        parser.error("counts must be positive and confidence must be between zero and one")
    results_path, summary_path = experiment_paths(args.output_suffix)
    if args.analyze_only:
        if args.analyze_only.resolve() == summary_path.resolve():
            parser.error("summary path must differ from input")
        write_summary(summary_path, load_jsonl(args.analyze_only), args)
        return
    if args.input.resolve() in (results_path.resolve(), summary_path.resolve()):
        parser.error("output paths must differ from input")
    sources = load_jsonl(args.input)[: args.max_examples]
    validate_sources(sources)
    model = args.model or (OPENAI_MODEL if args.llm == "openai" else OLLAMA_MODEL)
    modules = build_modules()
    prompt_hashes = {
        mode: hashlib.sha256(json.dumps({
            "instructions": module.generate.signature.instructions,
            "schema": module.generate.signature.model_json_schema(),
            "context_guard": guarded_answer_context("{context}"),
        }, sort_keys=True).encode()).hexdigest()
        for mode, module in modules.items()
    }
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "input_name": args.input.name,
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "examples": len(sources),
        "backend": args.llm,
        "model": model,
        "ollama_api_base": args.ollama_api_base if args.llm == "ollama" else None,
        "temperature": 0,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "lm_cache": False,
        "dspy_version": version("dspy"),
        "prompt_sha256": prompt_hashes,
    }
    backend = build_llm_backend(
        args.llm, model, openai_api_key=OPENAI_API_KEY,
        ollama_api_base=args.ollama_api_base, max_tokens=args.max_tokens, cache=False,
    )
    rows = prepare_output(results_path, sources, config, resume=args.resume)
    if len(rows) < len(sources):
        lm = backend.create_lm(temperature=0, seed=args.seed)
        with dspy.context(lm=lm):
            for source in tqdm(sources[len(rows):], desc="Answer ablation"):
                row = run_example(source, modules, [lm], seed=args.seed)
                row["run_config"] = config
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, allow_nan=False) + "\n")
                rows.append(row)
    write_summary(summary_path, rows, args)


if __name__ == "__main__":
    main()
