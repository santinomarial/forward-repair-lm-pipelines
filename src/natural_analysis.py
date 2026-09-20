"""Locked paired analysis of one fresh natural-error holdout; no model calls."""

from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from case_explorer import format_candidate
from data_loader import load_jsonl
from natural_routing import NATURAL_ACTIONS, NaturalRouter
from natural_study import model_lock, validate_rows, verify_inputs, write_new, write_rows_new
from outcome_routing import OutcomeRouter
from reliability_analysis import ClusterBootstrap
from reliability_study import context_for, digest, score


def paired_comparison(rows: list[dict], choices_a: list[str], choices_b: list[str], bootstrap) -> dict:
    initial = np.asarray([r["initial"]["metrics"]["exact_match"] for r in rows])
    outcomes = [[r["outcomes"][a] for r, a in zip(rows, choices)] for choices in (choices_a, choices_b)]
    em = [np.asarray([o["metrics"]["exact_match"] for o in selected]) for selected in outcomes]
    f1 = [np.asarray([o["metrics"]["token_f1"] for o in selected]) for selected in outcomes]
    delta = em[0] - em[1]
    return {"exact_match": bootstrap.ratio(delta), "token_f1": bootstrap.ratio(f1[0] - f1[1]),
            "recovery": bootstrap.ratio(delta * (1 - initial), 1 - initial),
            "damage": bootstrap.ratio(-delta * initial, initial),
            "a_only_correct": int((delta == 1).sum()), "b_only_correct": int((delta == -1).sum())}


def policy_metrics(rows: list[dict], choices: list[str], bootstrap) -> dict:
    initial = np.asarray([r["initial"]["metrics"]["exact_match"] for r in rows])
    selected = [r["outcomes"][a] for r, a in zip(rows, choices)]
    em = np.asarray([o["metrics"]["exact_match"] for o in selected])
    costs = [o["telemetry"]["estimated_cost_usd"] for o in selected]
    telemetry = [o["telemetry"] for o in selected]
    baselines = [r["baseline_telemetry"] for r in rows]
    return {
        "exact_match": bootstrap.ratio(em),
        "token_f1": bootstrap.ratio([o["metrics"]["token_f1"] for o in selected]),
        "recovery": bootstrap.ratio(em * (1 - initial), 1 - initial),
        "damage": bootstrap.ratio((1 - em) * initial, initial),
        "recovered_count": int((em * (1 - initial)).sum()),
        "damaged_count": int(((1 - em) * initial).sum()),
        "abstention_rate": float(np.mean([o["answer"].strip().rstrip(".").upper() == "UNKNOWN" for o in selected])),
        "recoveries_with_gold_phrase_already_present": sum(
            int(not r["initial"]["metrics"]["exact_match"] and o["metrics"]["exact_match"]
                and format_candidate(r["initial"]["answer"], r["gold"])) for r, o in zip(rows, selected)),
        "action_counts": dict(Counter(choices)),
        "incremental": {
            "calls_per_question": float(np.mean([t["llm_calls"] for t in telemetry])),
            "tokens_per_question": float(np.mean([t["total_tokens"] for t in telemetry])),
            "cost_usd_per_question": bootstrap.ratio(costs),
            "wall_seconds_mean": float(np.mean([t["wall_clock_seconds"] for t in telemetry])),
            "wall_seconds_p95": float(np.quantile([t["wall_clock_seconds"] for t in telemetry], .95)),
            "throttle_seconds_mean": float(np.mean([t.get("throttle_seconds", 0) for t in telemetry])),
            "unthrottled_seconds_mean": float(np.mean([max(0, t["wall_clock_seconds"] - t.get("throttle_seconds", 0)) for t in telemetry])),
            "stage_seconds_mean": {stage: float(np.mean([t["latency_seconds"][stage] for t in telemetry]))
                                   for stage in ("query_generation", "retrieval", "answer_generation")},
        },
        "total_including_baseline": {
            "calls_per_question": float(np.mean([a["llm_calls"] + b["llm_calls"] for a, b in zip(telemetry, baselines)])),
            "cost_usd_per_question": float(np.mean([a["estimated_cost_usd"] + b["estimated_cost_usd"] for a, b in zip(telemetry, baselines)])),
            "wall_seconds_mean": float(np.mean([a["wall_clock_seconds"] + b["wall_clock_seconds"] for a, b in zip(telemetry, baselines)])),
        },
    }


def verify_evidence(directory: Path, rows: list[dict]) -> None:
    corpus = {d["id"]: d for d in load_jsonl(directory / "corpus.jsonl")}
    for row in rows:
        initial = row["initial"]
        original = [d["id"] for d in initial["docs"]]
        if context_for([corpus[key] for key in original]) != initial["context"]:
            raise ValueError("initial evidence changed")
        for action, outcome in row["outcomes"].items():
            context = context_for([corpus[key] for key in outcome["doc_ids"]])
            if hashlib.sha256(context.encode()).hexdigest() != outcome["context_sha256"]:
                raise ValueError("action evidence hash mismatch")
            if action in ("accept", "regenerate") and outcome["doc_ids"] != original:
                raise ValueError("fixed-evidence action changed documents")
        if row["outcomes"]["accept"]["answer"] != initial["answer"]:
            raise ValueError("accept action changed the original answer")
        if not set(original).issubset(row["outcomes"]["preserve_evidence"]["doc_ids"]):
            raise ValueError("evidence-preserving action dropped original evidence")
        if len({row["outcomes"][a]["query"] for a in ("rewrite_query", "rewrite_query_10", "preserve_evidence")}) != 1:
            raise ValueError("rewrite arms used different queries")


def pilot_report(directory: Path) -> dict:
    manifest = verify_inputs(directory)
    if (directory / "test_model_lock.json").exists():
        raise ValueError("pilot analysis must precede held-out unlock")
    rows = load_jsonl(directory / "development_outcomes.jsonl")
    validate_rows(directory, rows, ("train", "validation"))
    rows = [r for r in rows if r["question_id"] in manifest["pilot_ids"]]
    if len(rows) != len(manifest["pilot_ids"]):
        raise ValueError("finish pilot before inspection")
    verify_evidence(directory, rows)
    initial = np.asarray([r["initial"]["metrics"]["exact_match"] for r in rows])
    ledger = load_jsonl(directory / "budget.jsonl")
    actions = {}
    for action in NATURAL_ACTIONS:
        em = np.asarray([r["outcomes"][action]["metrics"]["exact_match"] for r in rows])
        disagreement = float(np.mean(em != initial))
        actions[action] = {"em": float(em.mean()), "recovered": int(((1-initial)*em).sum()),
                           "damaged": int((initial*(1-em)).sum()), "discordance": disagreement,
                           "approx_95ci_halfwidth_at_n300_pp": 100 * 1.96 * np.sqrt(disagreement / 300)}
    result = {"questions": len(rows), "actions": actions,
              "reserved_usd": sum(r["reserved_usd"] for r in ledger),
              "completed_cost_usd": sum(r["collection_telemetry"]["estimated_cost_usd"] for r in rows),
              "heldout_questions": 300, "sample_size_changed": False,
              "note": "Pilot is training data, not confirmatory evidence. Approximate precision ignores effect mean; no power guarantee. Fixed 300-question test and no significance-based stopping.",
              "development_sha256": digest(directory / "development_outcomes.jsonl")}
    write_new(directory / "pilot_report.json", result)
    print(json.dumps(result, indent=2))
    return result


def markdown(report: dict) -> str:
    lines = ["# Natural-error repair: final held-out evaluation", "",
             f"{report['test_questions']} fresh held-out HotpotQA questions; BM25; gpt-4o-mini; no injected corruption.", "",
             "| Policy | EM | Recovery | Damage | Calls added | Cost added/question | Wall time added |",
             "|:--|--:|--:|--:|--:|--:|--:|"]
    for name, p in report["policies"].items():
        rates = ["—" if p[k]["estimate"] is None else f"{p[k]['estimate']:.1%}" for k in ("exact_match", "recovery", "damage")]
        t = p["incremental"]
        lines.append(f"| {name} | {' | '.join(rates)} | {t['calls_per_question']:.2f} | ${t['cost_usd_per_question']['estimate']:.6f} | {t['wall_seconds_mean']:.2f}s |")
    lines += ["", "## Paired differences", "", "The sole primary contrast is natural_damage_augmented minus accept; all others are exploratory.", "",
              "| Contrast | EM difference | 95% CI | A-only / B-only correct |", "|:--|--:|:--|--:|"]
    for name, values in report["comparisons"].items():
        em = values["exact_match"]
        lines.append(f"| {name} | {100*em['estimate']:+.1f} pp | [{100*em['ci'][0]:.1f}, {100*em['ci'][1]:.1f}] | {values['a_only_correct']} / {values['b_only_correct']} |")
    lines += ["", "## Limits", "", report["limitations"], "",
              "Costs are selected-policy estimates from offline observed actions, not an online deployment. "
              "Wall times include provider/local pacing and exclude router CPU; unthrottled and baseline-inclusive totals are in JSON. "
              "Shared query generation is charged once per selected rewrite policy but only once in actual collection cost.", "",
              "All EM-change cases for the primary policy are exported to review_queue.jsonl. "
              "Human semantic review remains pending; lexical improvement is not proof of factuality.", ""]
    return "\n".join(lines)


def create_report(directory: Path) -> dict:
    manifest = verify_inputs(directory)
    if json.loads((directory / "test_model_lock.json").read_text()) != model_lock(directory):
        raise ValueError("model/analysis changed after test unlock")
    phases = []
    for phase, splits in (("development", ("train", "validation")), ("test", ("test",))):
        rows = load_jsonl(directory / f"{phase}_outcomes.jsonl")
        validate_rows(directory, rows, splits)
        if len(rows) != sum(len(manifest["splits"][s]) for s in splits):
            raise ValueError("report requires complete development and test data")
        verify_evidence(directory, rows)
        phases.append(sorted(rows, key=lambda r: r["question_id"]))
    development, test = phases
    models = json.loads((directory / "models.json").read_text())["policies"]
    synthetic = OutcomeRouter.load(directory / "synthetic_model.json")
    choices = {action: [action] * len(test) for action in NATURAL_ACTIONS}
    choices["synthetic_original"] = [synthetic.decide(question=r["question"], initial=r["initial"]) for r in test]
    for name, payload in models.items():
        router = NaturalRouter(**payload["policy"])
        choices[name] = [router.decide(r) for r in test]
    bootstrap = ClusterBootstrap([r["question_id"] for r in test], resamples=manifest["resamples"], seed=41)
    contrasts = [("natural_damage_augmented", "accept"), ("preserve_evidence", "rewrite_query_10"),
                 ("preserve_evidence", "expand_context"), ("natural_original", "synthetic_original"),
                 ("natural_augmented", "natural_original"), ("natural_damage_augmented", "natural_augmented"),
                 ("natural_damage_augmented", "rewrite_query"), ("natural_damage_augmented", "expand_context")]
    ledger = load_jsonl(directory / "budget.jsonl")
    all_rows = development + test
    initial_wrong = sum(not r["initial"]["metrics"]["exact_match"] for r in test)
    result = {
        "complete": True, "test_questions": len(test), "development_questions": len(development),
        "initially_wrong": initial_wrong, "initially_correct": len(test) - initial_wrong,
        "manifest_sha256": digest(directory / "manifest.json"), "models_sha256": digest(directory / "models.json"),
        "test_sha256": digest(directory / "test_outcomes.jsonl"),
        "primary_comparison": manifest["primary_comparison"],
        "statistics": {"method": "paired question bootstrap", "resamples": 10000, "seed": 41,
                       "confidence": .95, "multiplicity_adjusted": False},
        "policies": {name: policy_metrics(test, actions, bootstrap) for name, actions in choices.items()},
        "comparisons": {f"{a}_minus_{b}": paired_comparison(test, choices[a], choices[b], bootstrap) for a, b in contrasts},
        "oracle": {"label": "Hindsight action-set ceiling, not a deployable policy",
                   "correct": sum(any(r["outcomes"][a]["metrics"]["exact_match"] for a in NATURAL_ACTIONS) for r in test),
                   "recoverable_initial_errors": sum(not r["initial"]["metrics"]["exact_match"] and any(
                       r["outcomes"][a]["metrics"]["exact_match"] for a in NATURAL_ACTIONS) for r in test)},
        "collection": {"completed_calls": sum(r["collection_telemetry"]["llm_calls"] for r in all_rows),
                       "completed_tokens": sum(r["collection_telemetry"]["total_tokens"] for r in all_rows),
                       "completed_cost_usd": sum(r["collection_telemetry"]["estimated_cost_usd"] for r in all_rows),
                       "reservation_cap_usd": 5, "reserved_usd": sum(r["reserved_usd"] for r in ledger),
                       "reservations": len(ledger), "note": "Reservations include failed/discarded requests; completed costs may exclude them."},
        "limitations": "One model seed, one benchmark, 300 held-out questions. New pooled corpus and bounded passages differ from the historical study, so raw EM is not a historical regression comparison. Equal token/document ceilings do not imply equal realized context length. Exact match uses the project's leading yes/no convention. Bootstrap intervals are pointwise, not multiplicity-adjusted; secondary contrasts are exploratory. Natural errors mean failures of unmodified benchmark outputs, not production incidents. No human semantic review is claimed.",
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (directory / "report.md").write_text(markdown(result))
    decisions = [{"question_id": r["question_id"], "choices": {name: actions[i] for name, actions in choices.items()}}
                 for i, r in enumerate(test)]
    # Derived files are safely reproducible; do not overwrite user-authored reviews.
    for name, rows in (("test_policy_choices.jsonl", decisions), ("review_queue.jsonl", [
        {"case_id": r["case_id"], "question": r["question"], "gold": r["gold"],
         "action": action, "before": r["initial"]["answer"], "after": r["outcomes"][action]["answer"],
         "before_doc_ids": [d["id"] for d in r["initial"]["docs"]], "after_doc_ids": r["outcomes"][action]["doc_ids"],
         "support_doc_ids": r["support_doc_ids"], "status": "pending", "reviewer": "", "labels": [], "rationale": ""}
        for r, action in zip(test, choices["natural_damage_augmented"])
        if score(r["initial"]["answer"], r["gold"])["exact_match"] != r["outcomes"][action]["metrics"]["exact_match"]
    ])):
        with (directory / name).open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps({"primary": result["comparisons"][manifest["primary_comparison"]], "collection": result["collection"]}, indent=2))
    return result
