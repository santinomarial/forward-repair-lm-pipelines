"""Collect action outcomes, freeze an outcome router, then evaluate held-out failures."""

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import copy
import fcntl
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import random
import re
from time import perf_counter

import dspy
from tqdm import tqdm

from config import (
    CORPUS_PATH, FIGURE_ANSWER_PATH, FIGURE_QUERY_SEED_PATHS,
    OPENAI_API_KEY, RELIABILITY_STUDY_DIR,
)
from data_loader import load_jsonl
from dspy_modules import AnswerGenerator, QueryGenerator, guarded_answer_context
from experiment_budget import BudgetedLM, ExperimentBudget
from metrics import contains_answer, exact_match, token_f1
from outcome_routing import (
    ACTIONS, SEEN_FAMILIES, UNSEEN_FAMILIES, OutcomeRouter, runtime_features, select_router,
)
from retriever import BM25Retriever
from telemetry import LMUsageSnapshot


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def context_for(docs: list[dict]) -> str:
    return "\n".join(f"Title: {doc['title']}\nText: {doc['text']}" for doc in docs)


def score(answer: str, gold: str) -> dict:
    return {"exact_match": exact_match(answer, gold), "token_f1": token_f1(answer, gold),
            "contains_answer": contains_answer(answer, gold)}


def zero_telemetry() -> dict:
    return {
        "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "estimated_cost_usd": 0.0, "priced_calls": 0, "estimated_token_calls": 0,
        "wall_clock_seconds": 0.0,
        "latency_seconds": {"query_generation": 0.0, "retrieval": 0.0, "answer_generation": 0.0},
    }


def question_splits(ids: list[str], seed: int = 17) -> dict[str, list[str]]:
    if len(set(ids)) != len(ids) or len(ids) < 5:
        raise ValueError("need at least five unique question IDs")
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    train_end, validation_end = int(n * 0.6), int(n * 0.8)
    return {"train": sorted(ordered[:train_end]), "validation": sorted(ordered[train_end:validation_end]),
            "test": sorted(ordered[validation_end:])}


def make_manifest(query_rows: list[dict], answer_rows: list[dict], sources: dict) -> dict:
    q = {str(r["id"]): r for r in query_rows}
    a = {str(r["id"]): r for r in answer_rows}
    if len(q) != len(query_rows) or len(a) != len(answer_rows) or q.keys() != a.keys():
        raise ValueError("source files need matching, unique question IDs")
    if any(q[i]["question"] != a[i]["question"] or q[i]["gold"] != a[i]["gold"] for i in q):
        raise ValueError("question/gold mismatch between source files")
    signatures = [AnswerGenerator().generate.signature, QueryGenerator("repaired").generate.signature]
    prompt_hash = hashlib.sha256(json.dumps({
        "signatures": [{"schema": s.model_json_schema(), "instructions": s.instructions} for s in signatures],
        "guard": guarded_answer_context("{context}"),
    }, sort_keys=True).encode()).hexdigest()
    return {
        "protocol_version": 1, "seed": 17, "provider_seed": 0, "model": "gpt-4o-mini",
        "dspy_version": version("dspy"), "sources": sources, "prompt_sha256": prompt_hash,
        "splits": question_splits(list(q)), "seen_families": list(SEEN_FAMILIES),
        "unseen_families": list(UNSEEN_FAMILIES), "actions": list(ACTIONS),
        "top_k": 5, "expanded_k": 10, "dropout_count": 2, "max_tokens": 300,
        "temperature": 0, "lm_cache": False, "cost_penalty": 100.0,
        "ridge_grid": [0.1, 1.0, 10.0, 100.0], "min_gain_grid": [0.0, 0.05, 0.1],
        "primary_metric": "exact_match", "bootstrap": "paired cluster bootstrap by question ID",
        "confidence": 0.95, "resamples": 10000,
    }


def make_cases(manifest: dict, phase: str) -> list[dict]:
    if phase not in ("development", "test"):
        raise ValueError("unknown collection phase")
    cases = []
    for split in (("train", "validation") if phase == "development" else ("test",)):
        families = SEEN_FAMILIES if split != "test" else (*SEEN_FAMILIES, *UNSEEN_FAMILIES, "natural")
        for question_id in manifest["splits"][split]:
            for family in families:
                cases.append({"case_id": f"{question_id}:{family}", "question_id": question_id,
                              "split": split, "family": family})
    return cases


def substitute_query(query: str, docs: list[dict], corpus: list[dict], case_id: str) -> tuple[str, dict]:
    """A deterministic unseen perturbation; uses no gold/support annotations."""
    words = re.findall(r"[A-Za-z]{4,}", query)
    if not words:
        raise ValueError("cannot substitute a query without an alphabetic term")
    term = max(words, key=len)
    excluded = {doc["id"] for doc in docs}
    candidates = [doc for doc in corpus if doc["id"] not in excluded and term.lower() not in doc["title"].lower()]
    if not candidates:
        raise ValueError("no replacement title available")
    replacement = random.Random(case_id).choice(candidates)["title"]
    corrupted = re.sub(rf"\b{re.escape(term)}\b", lambda _: replacement, query, count=1)
    if corrupted == query:
        raise ValueError("query substitution must change the input")
    return corrupted, {"removed_term": term, "replacement_title": replacement}


def run_case(case: dict, source: dict, corpus: list[dict], retriever, lm) -> dict:
    """Observe all actions for one state; inference features never receive labels."""
    question, gold = source["question"], source["gold"]
    answerer, rewriter = AnswerGenerator(), QueryGenerator("repaired")
    initial = copy.deepcopy(source["run"])
    setup_usage = LMUsageSnapshot([lm])
    setup_throttle = getattr(lm, "throttle_seconds", 0.0)
    setup_start = perf_counter()
    mutation: dict = {}
    family = case["family"]
    if family == "query_term_substitution":
        initial["query"], mutation = substitute_query(initial["query"], initial["docs"], corpus, case["case_id"])
        initial["docs"] = retriever.retrieve(initial["query"], top_k=5)
    elif family == "retrieval_rank_dropout":
        if len(initial["docs"]) <= 2:
            raise ValueError("dropout requires more than two original documents")
        mutation = {"removed_doc_ids": [doc["id"] for doc in initial["docs"][:2]]}
        initial["docs"] = initial["docs"][2:]
    if family in UNSEEN_FAMILIES:
        initial["context"] = context_for(initial["docs"])
        initial["answer"] = answerer(question=question, context=initial["context"]).answer
    initial = {key: initial[key] for key in ("query", "docs", "context", "answer")}
    setup = {**setup_usage.finish(), "wall_clock_seconds": perf_counter() - setup_start,
             "throttle_seconds": getattr(lm, "throttle_seconds", 0.0) - setup_throttle}
    initial["metrics"] = score(initial["answer"], gold)
    initial["origin"] = "new_perturbation" if family in UNSEEN_FAMILIES else "saved_source_run"
    result = {**case, "question": question, "gold": gold, "initial": initial,
              "mutation": mutation, "setup_telemetry": setup,
              "features": runtime_features(question, initial), "outcomes": {}}
    result["outcomes"]["accept"] = {
        "answer": initial["answer"], "metrics": initial["metrics"], "telemetry": zero_telemetry(),
        "query": initial["query"], "doc_ids": [doc["id"] for doc in initial["docs"]],
    }
    order = list(ACTIONS[1:])
    random.Random(case["case_id"]).shuffle(order)
    result["action_order"] = order
    for action in order:
        usage = LMUsageSnapshot([lm])
        throttle_started = getattr(lm, "throttle_seconds", 0.0)
        started = perf_counter()
        query, docs, context = initial["query"], initial["docs"], initial["context"]
        query_time = retrieval_time = 0.0
        if action == "rewrite_query":
            stage_start = perf_counter()
            query = rewriter(question=question, bad_query=query).query
            query_time = perf_counter() - stage_start
        if action in ("rewrite_query", "expand_context"):
            stage_start = perf_counter()
            docs = retriever.retrieve(query, top_k=10 if action == "expand_context" else 5)
            context = context_for(docs)
            retrieval_time = perf_counter() - stage_start
        stage_start = perf_counter()
        answer = answerer(question=question, context=context).answer
        answer_time = perf_counter() - stage_start
        telemetry = {**usage.finish(), "wall_clock_seconds": perf_counter() - started,
                     "throttle_seconds": getattr(lm, "throttle_seconds", 0.0) - throttle_started,
                     "latency_seconds": {"query_generation": query_time, "retrieval": retrieval_time,
                                         "answer_generation": answer_time}}
        if telemetry["priced_calls"] != telemetry["llm_calls"]:
            raise RuntimeError("missing cost data: refusing to train on unpriced outcomes")
        result["outcomes"][action] = {
            "answer": answer, "metrics": score(answer, gold), "telemetry": telemetry,
            "query": query, "doc_ids": [doc["id"] for doc in docs],
            "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
        }
    return result


def validate_records(rows: list[dict], cases: list[dict], manifest_sha: str) -> None:
    expected = {case["case_id"]: case for case in cases}
    seen = set()
    for row in rows:
        key = row["case_id"]
        if key in seen or key not in expected or row.get("manifest_sha256") != manifest_sha:
            raise ValueError("duplicate, unexpected, or incompatible saved case")
        if any(row[field] != expected[key][field] for field in ("question_id", "split", "family")):
            raise ValueError("saved split/family differs from manifest")
        if set(row["outcomes"]) != set(ACTIONS):
            raise ValueError("saved case lacks required action outcomes")
        seen.add(key)


@contextmanager
def study_lock(directory: Path):
    """Prevent independent collectors from racing the same reservation ledger."""
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".collection.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another process is using this study directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def collect(directory: Path, *, phase: str, workers: int, budget_usd: float, resume: bool, limit: int | None):
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")
    query_rows, answer_rows = load_jsonl(FIGURE_QUERY_SEED_PATHS[0]), load_jsonl(FIGURE_ANSWER_PATH)
    corpus = load_jsonl(CORPUS_PATH)
    paths = {"query": FIGURE_QUERY_SEED_PATHS[0], "answer": FIGURE_ANSWER_PATH, "corpus": CORPUS_PATH}
    manifest = make_manifest(query_rows, answer_rows, {
        name: {"name": path.name, "sha256": digest(path)} for name, path in paths.items()
    })
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("source files, settings, or prompts changed; use a separate study directory")
    else:
        write_new(manifest_path, manifest)
    manifest_sha = digest(manifest_path)
    if phase == "test":
        model_path = directory / "model.json"
        payload = json.loads(model_path.read_text())
        OutcomeRouter.load(model_path)
        if payload["metadata"]["manifest_sha256"] != manifest_sha:
            raise ValueError("model was fitted under a different manifest")
        if payload["metadata"]["development_sha256"] != digest(directory / "development_outcomes.jsonl"):
            raise ValueError("development data changed after model fit")
        lock_value = {"model_sha256": digest(model_path), "manifest_sha256": manifest_sha}
        lock_path = directory / "test_model_lock.json"
        if lock_path.exists():
            if json.loads(lock_path.read_text()) != lock_value:
                raise ValueError("model changed after test evaluation began")
        else:
            write_new(lock_path, lock_value)
    cases = make_cases(manifest, phase)
    output = directory / f"{phase}_outcomes.jsonl"
    if output.exists() and not resume:
        raise FileExistsError("outcomes already exist; pass --resume")
    rows = load_jsonl(output) if output.exists() else []
    validate_records(rows, cases, manifest_sha)
    done = {row["case_id"] for row in rows}
    remaining = [case for case in cases if case["case_id"] not in done]
    if limit is not None:
        remaining = remaining[:limit]
    q = {str(row["id"]): row for row in query_rows}
    a = {str(row["id"]): row for row in answer_rows}
    retriever = BM25Retriever(corpus, top_k=5)
    budget = ExperimentBudget(directory / "budget.jsonl", budget_usd)

    def worker(case):
        source_row = a[case["question_id"]] if case["family"] == "ignore_context" else q[case["question_id"]]
        mode = "corrupted" if case["family"] in SEEN_FAMILIES else "baseline"
        source = {"question": source_row["question"], "gold": source_row["gold"], "run": source_row[mode]}
        lm = BudgetedLM(budget, case["case_id"], api_key=OPENAI_API_KEY, seed=0)
        with dspy.context(lm=lm):
            row = run_case(case, source, corpus, retriever, lm)
        row["manifest_sha256"] = manifest_sha
        return row

    # Bound in-flight requests; stop submissions after an error but save completed peers.
    failures = []
    iterator = iter(remaining)
    with ThreadPoolExecutor(max_workers=workers) as executor, output.open("a", encoding="utf-8") as handle:
        pending = {executor.submit(worker, case) for case in [next(iterator, None) for _ in range(workers)] if case}
        with tqdm(total=len(remaining), desc=f"{phase} cases") as progress:
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    try:
                        row = future.result()
                        handle.write(json.dumps(row, allow_nan=False) + "\n")
                        handle.flush()
                        progress.update(1)
                    except Exception as exc:
                        failures.append(exc)
                if not failures:
                    for _ in completed:
                        case = next(iterator, None)
                        if case is not None:
                            pending.add(executor.submit(worker, case))
    print(f"Reserved upper budget: ${budget.reserved_usd:.4f} / ${budget.limit_usd:.2f}; requests: {budget.requests}")
    if failures:
        raise RuntimeError("collection stopped; completed cases and budget reservations were preserved") from failures[0]


def fit(directory: Path):
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = load_jsonl(directory / "development_outcomes.jsonl")
    cases = make_cases(manifest, "development")
    validate_records(rows, cases, digest(manifest_path))
    if len(rows) != len(cases):
        raise ValueError("finish all development cases before fitting")
    if (directory / "test_model_lock.json").exists():
        raise ValueError("cannot fit after test outcomes have been unlocked")
    # Sort so completion order cannot affect floating-point fitting or selection.
    rows.sort(key=lambda row: row["case_id"])
    train = [row for row in rows if row["split"] == "train"]
    validation = [row for row in rows if row["split"] == "validation"]
    policy, selection = select_router(train, validation, cost_penalty=manifest["cost_penalty"])
    metadata = {"manifest_sha256": digest(manifest_path), "development_sha256": digest(directory / "development_outcomes.jsonl"),
                "training_questions": len(manifest["splits"]["train"]),
                "validation_questions": len(manifest["splits"]["validation"]), "selection": selection}
    policy.save(directory / "model.json", metadata)
    print(json.dumps({"selected": selection["candidates"][selection["selected_index"]]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["collect", "fit", "report"])
    parser.add_argument("--directory", type=Path, default=RELIABILITY_STUDY_DIR)
    parser.add_argument("--phase", choices=["development", "test"], default="development")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--budget-usd", type=float, default=3.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, help="Collect at most this many new cases; cannot fit partial data.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or not 0 < args.budget_usd <= 3 or (args.limit is not None and args.limit <= 0):
        parser.error("use 1–3 workers, a budget in (0, 3], and a positive case limit")
    with study_lock(args.directory):
        if args.command == "collect":
            collect(args.directory, phase=args.phase, workers=args.workers, budget_usd=args.budget_usd,
                    resume=args.resume, limit=args.limit)
        elif args.command == "fit":
            fit(args.directory)
        else:
            from reliability_analysis import create_report
            create_report(args.directory)


if __name__ == "__main__":
    main()
