"""Read-only replay of frozen study cases; never constructs or calls an LM."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from config import CORPUS_PATH, EXAMPLES_PATH, RELIABILITY_STUDY_DIR
from data_loader import load_jsonl
from metrics import exact_match, normalize
from outcome_routing import ACTIONS, OUTCOME_FEATURES, OutcomeRouter, runtime_features
from reliability_analysis import read_frozen_study
from reliability_study import context_for, digest, score


ACTION_LABELS = {
    "accept": "Keep original", "rewrite_query": "Rewrite query",
    "expand_context": "Retrieve more", "regenerate": "Fresh answer",
}
FAMILY_LABELS = {
    "vague_query": "Vague query", "ignore_context": "Ignored context",
    "query_term_substitution": "Query-term substitution",
    "retrieval_rank_dropout": "Retrieval dropout", "natural": "Natural output",
}
TRANSITIONS = {(0, 1): "Recovered", (1, 0): "Harmed", (1, 1): "Still exact match", (0, 0): "Still not exact match"}


def phrase_present(phrase: str, text: str) -> bool:
    """Whole normalized phrase, not substring matching (Ann != Anne)."""
    phrase = normalize(phrase)
    return bool(phrase) and f" {phrase} " in f" {normalize(text)} "


def format_candidate(answer: str, gold: str) -> bool:
    return (normalize(gold) not in ("yes", "no", "")
            and not exact_match(answer, gold) and phrase_present(gold, answer))


def evidence_for(row: dict, action: str, corpus: dict[str, dict]) -> list[dict]:
    if action not in ACTIONS:
        raise ValueError("unknown action")
    if action in ("accept", "regenerate"):
        docs = row["initial"]["docs"]
        context = row["initial"]["context"]
        if context_for(docs) != context or any(
            doc["id"] not in corpus or any(doc[key] != corpus[doc["id"]][key] for key in ("title", "text"))
            for doc in docs
        ):
            raise ValueError("initial evidence text differs from the saved context or corpus")
    else:
        try:
            docs = [corpus[key] for key in row["outcomes"][action]["doc_ids"]]
        except KeyError as exc:
            raise ValueError("saved document is missing from the corpus") from exc
        context = context_for(docs)
    if [doc["id"] for doc in docs] != row["outcomes"][action]["doc_ids"]:
        raise ValueError("document order differs from the saved outcome")
    if action != "accept" and hashlib.sha256(context.encode()).hexdigest() != row["outcomes"][action]["context_sha256"]:
        raise ValueError("reconstructed evidence does not match the saved context hash")
    return docs


def annotate_case(row: dict, action: str, support_ids: list[str]) -> dict:
    """Gold-dependent audit only. These fields must never become router features."""
    before = score(row["initial"]["answer"], row["gold"])
    after = score(row["outcomes"][action]["answer"], row["gold"])
    before_ids = {doc["id"] for doc in row["initial"]["docs"]}
    after_ids = set(row["outcomes"][action]["doc_ids"])
    supports = set(support_ids)
    flags = []
    for name, answer, metrics, ids in (
        ("before", row["initial"]["answer"], before, before_ids),
        ("after", row["outcomes"][action]["answer"], after, after_ids),
    ):
        if format_candidate(answer, row["gold"]):
            flags.append(f"{name}: gold phrase present despite EM failure")
        if supports - ids:
            flags.append(f"{name}: annotated support missing")
        elif supports and not metrics["exact_match"]:
            flags.append(f"{name}: EM failure with all annotated support")
    return {
        "case_id": row["case_id"], "question_id": row["question_id"], "family": row["family"],
        "question": row["question"], "action": action,
        "transition": TRANSITIONS[(before["exact_match"], after["exact_match"])],
        "before_metrics": before, "after_metrics": after, "flags": flags,
        "support_doc_ids": sorted(supports),
        "missing_support_before": sorted(supports - before_ids),
        "missing_support_after": sorted(supports - after_ids),
        "kept_correct": action == "accept" and bool(before["exact_match"]),
        "missed_recovery": not after["exact_match"] and any(
            exact_match(row["outcomes"][a]["answer"], row["gold"]) for a in ACTIONS
        ),
        "format_candidate": format_candidate(row["initial"]["answer"], row["gold"])
                            or format_candidate(row["outcomes"][action]["answer"], row["gold"]),
        "recovery_with_gold_already_present": not before["exact_match"] and bool(after["exact_match"])
                                             and format_candidate(row["initial"]["answer"], row["gold"]),
    }


@dataclass
class Snapshot:
    rows: dict[str, dict]
    cases: list[dict]
    corpus: dict[str, dict]
    policy: OutcomeRouter
    provenance: dict


def load_snapshot(directory: Path = RELIABILITY_STUDY_DIR, corpus_path: Path = CORPUS_PATH,
                  examples_path: Path = EXAMPLES_PATH) -> Snapshot:
    manifest, _, rows, policy = read_frozen_study(directory)
    if digest(corpus_path) != manifest["sources"]["corpus"]["sha256"]:
        raise ValueError("corpus differs from the frozen study")
    corpus_rows, examples = load_jsonl(corpus_path), load_jsonl(examples_path)
    corpus = {doc["id"]: doc for doc in corpus_rows}
    annotations = {str(example["id"]): example for example in examples}
    if len(corpus) != len(corpus_rows) or len(annotations) != len(examples):
        raise ValueError("duplicate corpus or example IDs")
    cases = []
    for row in rows:
        example = annotations.get(row["question_id"])
        if example is None or example["question"] != row["question"] or example["answer"] != row["gold"]:
            raise ValueError("benchmark annotations do not match the saved question and gold")
        if not set(example["support_doc_ids"]).issubset(corpus):
            raise ValueError("annotated support missing from corpus")
        features = runtime_features(row["question"], row["initial"])
        if not np.allclose(features, row["features"], rtol=0, atol=1e-10):
            raise ValueError("runtime features differ from saved features")
        if row["outcomes"]["accept"]["answer"] != row["initial"]["answer"]:
            raise ValueError("accept must preserve the original answer")
        for action in ACTIONS:
            evidence_for(row, action, corpus)
            if score(row["outcomes"][action]["answer"], row["gold"]) != row["outcomes"][action]["metrics"]:
                raise ValueError("saved metrics differ from recomputed metrics")
        action = policy.decide_features(features)
        case = annotate_case(row, action, example["support_doc_ids"])
        case["predicted_utilities"] = dict(zip(ACTIONS, policy.utilities(features).tolist()))
        case["features"] = dict(zip(OUTCOME_FEATURES, features))
        cases.append(case)
    provenance = {path.name: digest(path) for path in (
        directory / "manifest.json", directory / "model.json", directory / "test_outcomes.jsonl",
        corpus_path, examples_path,
    )}
    return Snapshot({r["case_id"]: r for r in rows}, cases, corpus, policy, provenance)


def filter_cases(cases: list[dict], *, family: str = "All", view: str = "All cases", search: str = "") -> list[dict]:
    if view not in ("All cases", "Recovered", "Harmed", "Kept correct", "Missed recovery", "Format candidates"):
        raise ValueError("unknown case view")
    return [case for case in cases
            if (family == "All" or case["family"] == family)
            and (not search.strip() or search.strip().casefold() in (case["question"] + " " + case["case_id"]).casefold())
            and (view == "All cases" or case["transition"] == view
                 or (view == "Kept correct" and case["kept_correct"])
                 or (view == "Missed recovery" and case["missed_recovery"])
                 or (view == "Format candidates" and case["format_candidate"]))]


def case_export(snapshot: Snapshot, case: dict) -> str:
    row = snapshot.rows[case["case_id"]]
    return json.dumps({"provenance": snapshot.provenance, "audit": case, "record": row,
                       "action_documents": {a: evidence_for(row, a, snapshot.corpus) for a in ACTIONS}},
                      indent=2, allow_nan=False)
