"""Bounded fresh-question natural-error study: prepare, collect, freeze, report."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import random
from time import perf_counter

import dspy
import tiktoken
from tqdm import tqdm

from config import EXAMPLES_PATH, NATURAL_STUDY_DIR, OPENAI_API_KEY, RELIABILITY_STUDY_DIR
from data_loader import load_jsonl
from dspy_modules import AnswerGenerator, QueryGenerator, guarded_answer_context
from experiment_budget import BudgetedLM, ExperimentBudget
from natural_routing import NATURAL_ACTIONS, select_natural_router
from outcome_routing import ACTIONS, runtime_features
from reliability_study import context_for, digest, score, study_lock, write_new, zero_telemetry
from retriever import BM25Retriever
from telemetry import LMUsageSnapshot


SEED = 41
CAP_USD = 5.0
COLLECTION_SOURCES = ("natural_study.py", "natural_routing.py", "experiment_budget.py",
                      "dspy_modules.py", "retriever.py", "telemetry.py", "metrics.py", "routing.py",
                      "outcome_routing.py", "reliability_study.py")


def code_hashes() -> dict:
    return {name: digest(Path(__file__).with_name(name)) for name in COLLECTION_SOURCES}


def write_rows_new(path: Path, rows: list[dict]) -> None:
    with path.open("x") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def prepare(directory: Path) -> None:
    from datasets import load_dataset

    if (directory / "manifest.json").exists():
        raise FileExistsError("study already prepared; immutable inputs must not be replaced")
    old = load_jsonl(EXAMPLES_PATH)
    excluded_ids = {r["id"] for r in old}
    excluded_questions = {r["question"].strip().casefold() for r in old}
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    indices = list(range(len(dataset)))
    random.Random(SEED).shuffle(indices)
    selected = []
    for index in indices:
        item = dataset[index]
        q = item["question"].strip().casefold()
        if item["id"] not in excluded_ids and q not in excluded_questions:
            selected.append(item)
            excluded_questions.add(q)
        if len(selected) == 600:
            break
    if len(selected) != 600:
        raise ValueError("not enough disjoint fresh questions")
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    documents: dict[str, dict] = {}
    examples = []
    for index, item in enumerate(selected):
        support_ids = []
        for title, sentences in zip(item["context"]["title"], item["context"]["sentences"]):
            # Exact title hashing avoids slug collisions across punctuation/case.
            doc_id = hashlib.sha256(title.encode()).hexdigest()[:24]
            if title in item["supporting_facts"]["title"]:
                support_ids.append(doc_id)
            if doc_id not in documents:
                text = " ".join(sentences)
                tokens = encoding.encode(text)
                limit = 256 - len(encoding.encode(f"Title: {title}\nText: ")) - 2
                if limit < 1:
                    raise ValueError("document title exceeds context budget")
                clipped = encoding.decode(tokens[:limit])
                doc = {"id": doc_id, "title": title, "text": clipped,
                       "truncated": len(tokens) > limit}
                if len(encoding.encode(context_for([doc]))) > 256:
                    raise ValueError("document token ceiling exceeded")
                documents[doc_id] = doc
        if not support_ids:
            raise ValueError("question without annotated support")
        examples.append({"id": item["id"], "question": item["question"], "answer": item["answer"],
                         "support_doc_ids": sorted(set(support_ids)),
                         "split": "train" if index < 240 else "validation" if index < 300 else "test"})
    directory.mkdir(parents=True, exist_ok=True)
    write_rows_new(directory / "examples.jsonl", examples)
    write_rows_new(directory / "corpus.jsonl", list(documents.values()))
    old_model = json.loads((RELIABILITY_STUDY_DIR / "model.json").read_text())
    write_new(directory / "synthetic_model.json", old_model)
    signatures = [QueryGenerator("baseline").generate.signature, QueryGenerator("repaired").generate.signature,
                  AnswerGenerator().generate.signature]
    prompts = {"signatures": [{"schema": s.model_json_schema(), "instructions": s.instructions} for s in signatures],
               "guard": guarded_answer_context("{context}")}
    manifest = {
        "protocol_version": 1, "seed": SEED, "provider_seed": 0, "model": "gpt-4o-mini",
        "dspy_version": version("dspy"), "dataset": "hotpotqa/hotpot_qa distractor validation",
        "dataset_fingerprint": dataset._fingerprint,
        "excluded_historical_examples_sha256": digest(EXAMPLES_PATH),
        "sources": {name: digest(directory / name) for name in ("examples.jsonl", "corpus.jsonl", "synthetic_model.json")},
        "code_sha256": code_hashes(), "prompt_sha256": hashlib.sha256(json.dumps(prompts, sort_keys=True).encode()).hexdigest(),
        "splits": {split: [r["id"] for r in examples if r["split"] == split] for split in ("train", "validation", "test")},
        "pilot_ids": [r["id"] for r in examples[:60]], "budget_cap_usd": CAP_USD,
        "actions": list(NATURAL_ACTIONS), "document_token_limit": 256, "context_token_limit": 3000,
        "top_k": 5, "expanded_k": 10, "max_output_tokens": 300, "temperature": 0, "cache": False,
        "primary_comparison": "natural_damage_augmented_minus_accept",
        "primary_metric": "exact_match", "resamples": 10000,
        "policy_grid": {"ridge": [.1, 1, 10, 100], "min_gain": [0, .05, .1], "cost_penalty": 100,
                        "damage_penalty": [0, 1]},
        "stop_rule": "600 questions once, or reservation cap/provider failure; no outcome-driven sample expansion",
        "context_note": "New pooled corpus; equal per-document and total token ceilings, not equal realized token counts. No historical EM regression comparison.",
    }
    write_new(directory / "manifest.json", manifest)
    print(f"Prepared 600 disjoint questions and {len(documents)} documents; pilot=60, train=240, validation=60, test=300.")


def verify_inputs(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["code_sha256"] != code_hashes():
        raise ValueError("collection code changed after preparation")
    if manifest["dspy_version"] != version("dspy") or manifest["budget_cap_usd"] != CAP_USD:
        raise ValueError("runtime version or budget changed")
    for name, expected in manifest["sources"].items():
        if digest(directory / name) != expected:
            raise ValueError(f"frozen input changed: {name}")
    return manifest


def preserve_documents(original: list[dict], rewritten: list[dict], limit: int = 10) -> list[dict]:
    if len(original) > limit:
        raise ValueError("cannot preserve all original documents within limit")
    result: list[dict] = []
    seen = set()
    for doc in original + rewritten:
        if doc["id"] not in seen and len(result) < limit:
            result.append(dict(doc))
            seen.add(doc["id"])
    return result


def combine_telemetry(*items: dict) -> dict:
    result = zero_telemetry()
    result["throttle_seconds"] = 0.0
    for item in items:
        for key in result:
            if key == "latency_seconds":
                for stage in result[key]:
                    result[key][stage] += item[key][stage]
            else:
                result[key] += item.get(key, 0)
    return result


def measured(lm, stage: str, function):
    usage = LMUsageSnapshot([lm])
    throttle = getattr(lm, "throttle_seconds", 0.0)
    start = perf_counter()
    value = function()
    elapsed = perf_counter() - start
    telemetry = {**zero_telemetry(), **usage.finish(), "wall_clock_seconds": elapsed,
                 "throttle_seconds": getattr(lm, "throttle_seconds", 0.0) - throttle}
    telemetry["latency_seconds"][stage] = elapsed
    if telemetry["llm_calls"] != telemetry["priced_calls"]:
        raise RuntimeError("unpriced model request")
    return value, telemetry


def run_question(example: dict, retriever, lm) -> dict:
    question, gold = example["question"], example["answer"]
    answerer = AnswerGenerator()
    encoding = tiktoken.encoding_for_model("gpt-4o-mini")

    def generate(query, docs):
        context = context_for(docs)
        if len(encoding.encode(context)) > 3000:
            raise ValueError("context exceeds frozen token ceiling")
        result, telemetry = measured(lm, "answer_generation", lambda: answerer(question=question, context=context))
        return {"query": query, "doc_ids": [d["id"] for d in docs], "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                "context_tokens": len(encoding.encode(context)), "answer": result.answer,
                "metrics": score(result.answer, gold), "telemetry": telemetry}

    query_result, q_usage = measured(lm, "query_generation", lambda: QueryGenerator("baseline")(question=question))
    query = query_result.query
    docs, r_usage = measured(lm, "retrieval", lambda: retriever.retrieve(query, top_k=5))
    initial_outcome = generate(query, docs)
    baseline_usage = combine_telemetry(q_usage, r_usage, initial_outcome["telemetry"])
    initial = {"query": query, "docs": docs, "context": context_for(docs),
               "answer": initial_outcome["answer"], "metrics": initial_outcome["metrics"]}
    rewrite, rewrite_usage = measured(lm, "query_generation", lambda: QueryGenerator("repaired")(question=question, bad_query=query))
    # Share exactly one rewritten query across the three rewrite arms. Charge that
    # query once in actual collection, but once per selected rewrite policy too.
    order = list(NATURAL_ACTIONS[1:])
    random.Random(example["id"]).shuffle(order)
    outcomes = {"accept": {**initial_outcome, "telemetry": zero_telemetry()}}
    physical = [baseline_usage, rewrite_usage]
    for action in order:
        action_query = rewrite.query if action in ("rewrite_query", "rewrite_query_10", "preserve_evidence") else query
        action_docs, retrieval_usage = docs, zero_telemetry()
        if action != "regenerate":
            def retrieve():
                found = retriever.retrieve(action_query, top_k=5 if action == "rewrite_query" else 10)
                return preserve_documents(docs, found) if action == "preserve_evidence" else found
            action_docs, retrieval_usage = measured(lm, "retrieval", retrieve)
        outcome = generate(action_query, action_docs)
        own_usage = combine_telemetry(retrieval_usage, outcome["telemetry"])
        physical.append(own_usage)
        outcome["telemetry"] = combine_telemetry(own_usage, rewrite_usage) if action in (
            "rewrite_query", "rewrite_query_10", "preserve_evidence") else own_usage
        outcomes[action] = outcome
    return {"case_id": example["id"] + ":natural", "question_id": example["id"], "split": example["split"],
            "family": "natural", "question": question, "gold": gold, "initial": initial,
            "support_doc_ids": example["support_doc_ids"], "features": runtime_features(question, initial),
            "action_order": order, "outcomes": outcomes, "baseline_telemetry": baseline_usage,
            "shared_rewrite_telemetry": rewrite_usage, "collection_telemetry": combine_telemetry(*physical)}


def validate_rows(directory: Path, rows: list[dict], splits: tuple[str, ...]) -> None:
    examples = {r["id"]: r for r in load_jsonl(directory / "examples.jsonl") if r["split"] in splits}
    seen = set()
    for row in rows:
        key = row["question_id"]
        if key in seen or key not in examples or row["manifest_sha256"] != digest(directory / "manifest.json"):
            raise ValueError("duplicate, unexpected, or incompatible checkpoint")
        ex = examples[key]
        if (row["split"] != ex["split"] or row["question"] != ex["question"] or row["gold"] != ex["answer"]
                or row["family"] != "natural" or row["case_id"] != key + ":natural"
                or set(row["outcomes"]) != set(NATURAL_ACTIONS)):
            raise ValueError("checkpoint structure differs from frozen inputs")
        if row["features"] != runtime_features(row["question"], row["initial"]):
            raise ValueError("checkpoint features differ from inference whitelist")
        for outcome in [row["initial"], *row["outcomes"].values()]:
            if outcome["metrics"] != score(outcome["answer"], row["gold"]):
                raise ValueError("checkpoint score mismatch")
        seen.add(key)


def model_lock(directory: Path) -> dict:
    models = json.loads((directory / "models.json").read_text())
    if models["development_sha256"] != digest(directory / "development_outcomes.jsonl"):
        raise ValueError("development data changed after fit")
    if models["manifest_sha256"] != digest(directory / "manifest.json"):
        raise ValueError("model manifest mismatch")
    return {"models_sha256": digest(directory / "models.json"), "manifest_sha256": digest(directory / "manifest.json"),
            "analysis_sha256": digest(Path(__file__).with_name("natural_analysis.py"))}


def collect(directory: Path, *, phase: str, workers: int = 3, resume: bool = False, limit: int | None = None) -> None:
    manifest = verify_inputs(directory)
    if phase not in ("pilot", "development", "test") or not 1 <= workers <= 3:
        raise ValueError("invalid phase or worker count")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    lock_path = directory / "test_model_lock.json"
    if phase != "test" and lock_path.exists():
        raise ValueError("development collection forbidden after test unlock")
    if phase == "test":
        lock = model_lock(directory)
        if lock_path.exists():
            if json.loads(lock_path.read_text()) != lock:
                raise ValueError("frozen model or analysis changed after test unlock")
        else:
            write_new(lock_path, lock)
    splits = ("test",) if phase == "test" else ("train", "validation")
    examples = [r for r in load_jsonl(directory / "examples.jsonl") if r["split"] in splits]
    path = directory / ("test_outcomes.jsonl" if phase == "test" else "development_outcomes.jsonl")
    if path.exists() and not resume:
        raise FileExistsError("checkpoint exists; use --resume")
    saved = load_jsonl(path) if path.exists() else []
    validate_rows(directory, saved, splits)
    done = {r["question_id"] for r in saved}
    remaining = [r for r in examples if r["id"] not in done and (phase != "pilot" or r["id"] in manifest["pilot_ids"])]
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        remaining = remaining[:limit]
    budget = ExperimentBudget(directory / "budget.jsonl", CAP_USD)
    retriever = BM25Retriever(load_jsonl(directory / "corpus.jsonl"), top_k=5)

    def worker(example):
        lm = BudgetedLM(budget, example["id"], api_key=OPENAI_API_KEY, seed=0)
        with dspy.context(lm=lm):
            row = run_question(example, retriever, lm)
        row["manifest_sha256"] = digest(directory / "manifest.json")
        return row

    failures = []
    iterator = iter(remaining)
    with ThreadPoolExecutor(max_workers=workers) as executor, path.open("a") as handle:
        pending = {executor.submit(worker, r) for r in [next(iterator, None) for _ in range(workers)] if r}
        with tqdm(total=len(remaining), desc=phase) as progress:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    try:
                        row = future.result()
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                        progress.update(1)
                    except Exception as exc:
                        failures.append(exc)
                if not failures:
                    for _ in completed:
                        example = next(iterator, None)
                        if example is not None:
                            pending.add(executor.submit(worker, example))
    print(f"Reserved ${budget.reserved_usd:.4f} / ${CAP_USD:.2f}; requests={budget.requests}")
    if failures:
        raise RuntimeError("collection stopped; completed cases and all reservations preserved") from failures[0]


def fit(directory: Path) -> dict:
    manifest = verify_inputs(directory)
    if (directory / "test_model_lock.json").exists():
        raise ValueError("cannot fit after test unlock")
    rows = load_jsonl(directory / "development_outcomes.jsonl")
    validate_rows(directory, rows, ("train", "validation"))
    if len(rows) != len(manifest["splits"]["train"]) + len(manifest["splits"]["validation"]):
        raise ValueError("finish development before fitting")
    rows.sort(key=lambda r: r["question_id"])
    train = [r for r in rows if r["split"] == "train"]
    validation = [r for r in rows if r["split"] == "validation"]
    policies = {}
    for name, actions, penalty in (("natural_original", ACTIONS, 0.0),
                                   ("natural_augmented", NATURAL_ACTIONS, 0.0),
                                   ("natural_damage_augmented", NATURAL_ACTIONS, 1.0)):
        policy, selection = select_natural_router(train, validation, actions=actions, damage_penalty=penalty)
        policies[name] = {"policy": policy, "selection": selection}
    result = {"manifest_sha256": digest(directory / "manifest.json"),
              "development_sha256": digest(directory / "development_outcomes.jsonl"), "policies": policies}
    write_new(directory / "models.json", result)
    print(json.dumps({name: entry["selection"]["candidates"][entry["selection"]["selected_index"]]
                      for name, entry in policies.items()}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "collect", "fit", "report", "pilot-report"])
    parser.add_argument("--directory", type=Path, default=NATURAL_STUDY_DIR)
    parser.add_argument("--phase", choices=["pilot", "development", "test"], default="pilot")
    parser.add_argument("--workers", type=int, choices=[1, 2, 3], default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    with study_lock(args.directory):
        if args.command == "prepare":
            prepare(args.directory)
        elif args.command == "collect":
            collect(args.directory, phase=args.phase, workers=args.workers, resume=args.resume, limit=args.limit)
        elif args.command == "fit":
            fit(args.directory)
        else:
            from natural_analysis import create_report, pilot_report
            (pilot_report if args.command == "pilot-report" else create_report)(args.directory)


if __name__ == "__main__":
    main()
