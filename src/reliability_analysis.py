"""Offline paired, question-clustered evaluation of a frozen outcome router."""

from collections import Counter
import json
from pathlib import Path

import numpy as np

from data_loader import load_jsonl
from outcome_routing import ACTIONS, SEEN_FAMILIES, UNSEEN_FAMILIES, OutcomeRouter
from reliability_study import digest, make_cases, score, validate_records


class ClusterBootstrap:
    """Resample question IDs, retaining every variant and each paired policy."""

    def __init__(self, question_ids: list[str], *, resamples: int = 10000, seed: int = 17):
        if not question_ids or resamples <= 0:
            raise ValueError("bootstrap needs cases and positive resamples")
        unique = sorted(set(question_ids))
        self.groups = [np.flatnonzero(np.asarray(question_ids) == key) for key in unique]
        self.indices = np.random.default_rng(seed).integers(0, len(unique), (resamples, len(unique)))
        self.n = len(question_ids)

    def ratio(self, numerator, denominator=None) -> dict:
        num = np.asarray(numerator, dtype=float)
        den = np.ones(self.n) if denominator is None else np.asarray(denominator, dtype=float)
        if num.shape != (self.n,) or den.shape != (self.n,) or not np.isfinite(num).all() or not np.isfinite(den).all() or (den < 0).any():
            raise ValueError("invalid bootstrap arrays")
        if den.sum() == 0:
            return {"estimate": None, "ci": None, "eligible_count": 0, "valid_resamples": 0}
        group_num = np.asarray([num[group].sum() for group in self.groups])
        group_den = np.asarray([den[group].sum() for group in self.groups])
        sampled_num = group_num[self.indices].sum(axis=1)
        sampled_den = group_den[self.indices].sum(axis=1)
        valid = sampled_den > 0
        values = sampled_num[valid] / sampled_den[valid]
        return {
            "estimate": float(num.sum() / den.sum()),
            "ci": [float(x) for x in np.quantile(values, [.025, .975])],
            "eligible_count": int(den.sum()), "valid_resamples": int(valid.sum()),
        }


def evaluate_group(rows: list[dict], choices: dict[str, str], *, resamples: int = 10000) -> dict:
    if not rows:
        return {"cases": 0, "questions": 0, "policies": {}, "comparisons": {}}
    bootstrap = ClusterBootstrap([r["question_id"] for r in rows], resamples=resamples)
    initial = np.asarray([score(r["initial"]["answer"], r["gold"])["exact_match"] for r in rows])
    broken, correct = 1 - initial, initial
    result: dict = {
        "cases": len(rows), "questions": len({r["question_id"] for r in rows}),
        "initially_wrong": int(broken.sum()), "initially_correct": int(correct.sum()),
        "policies": {}, "comparisons": {},
    }
    outcomes_by_policy = {}
    for name in (*ACTIONS, "learned"):
        actions = [choices[r["case_id"]] if name == "learned" else name for r in rows]
        selected = [row["outcomes"][action] for row, action in zip(rows, actions)]
        scores = [score(item["answer"], row["gold"]) for row, item in zip(rows, selected)]
        em = np.asarray([s["exact_match"] for s in scores])
        f1 = np.asarray([s["token_f1"] for s in scores])
        telemetry = [item["telemetry"] for item in selected]
        costs = np.asarray([t["estimated_cost_usd"] for t in telemetry])
        latencies = np.asarray([t["wall_clock_seconds"] for t in telemetry])
        throttle = np.asarray([t.get("throttle_seconds", 0.0) for t in telemetry])
        result["policies"][name] = {
            "exact_match": bootstrap.ratio(em), "token_f1": bootstrap.ratio(f1),
            "contains_answer": float(np.mean([s["contains_answer"] for s in scores])),
            "recovery": bootstrap.ratio(em * broken, broken),
            "damage": bootstrap.ratio((1 - em) * correct, correct),
            "recovered_count": int((em * broken).sum()),
            "damaged_count": int(((1 - em) * correct).sum()),
            "utility": float(np.mean(em - 100 * costs)),
            "action_counts": dict(Counter(actions)),
            "incremental": {
                "calls_per_case": float(np.mean([t["llm_calls"] for t in telemetry])),
                "tokens_per_case": float(np.mean([t["total_tokens"] for t in telemetry])),
                "cost_usd_per_case": bootstrap.ratio(costs),
                "cost_usd_total": float(costs.sum()),
                "priced_calls": sum(t["priced_calls"] for t in telemetry),
                "latency_seconds_mean": float(latencies.mean()),
                "latency_seconds_p95": float(np.quantile(latencies, .95)),
                "throttle_seconds_mean": float(throttle.mean()),
                "unthrottled_seconds_mean": float(np.maximum(0, latencies - throttle).mean()),
                "stage_seconds_per_case": {
                    stage: float(np.mean([t["latency_seconds"][stage] for t in telemetry]))
                    for stage in ("query_generation", "retrieval", "answer_generation")
                },
            },
        }
        outcomes_by_policy[name] = (em, f1, costs)
    learned_em, learned_f1, learned_cost = outcomes_by_policy["learned"]
    for baseline in ACTIONS:
        em, f1, costs = outcomes_by_policy[baseline]
        result["comparisons"][f"learned_minus_{baseline}"] = {
            "exact_match": bootstrap.ratio(learned_em - em),
            "token_f1": bootstrap.ratio(learned_f1 - f1),
            "recovery": bootstrap.ratio((learned_em - em) * broken, broken),
            "damage": bootstrap.ratio((em - learned_em) * correct, correct),
            "cost_usd_per_case": bootstrap.ratio(learned_cost - costs),
        }
    return result


def read_frozen_study(directory: Path) -> tuple[dict, list[dict], list[dict], OutcomeRouter]:
    manifest_path, model_path = directory / "manifest.json", directory / "model.json"
    manifest = json.loads(manifest_path.read_text())
    lock = json.loads((directory / "test_model_lock.json").read_text())
    if lock != {"model_sha256": digest(model_path), "manifest_sha256": digest(manifest_path)}:
        raise ValueError("model or manifest changed after test collection began")
    metadata = json.loads(model_path.read_text())["metadata"]
    if metadata["development_sha256"] != digest(directory / "development_outcomes.jsonl"):
        raise ValueError("development observations changed after model fit")
    phases = []
    for phase in ("development", "test"):
        rows = load_jsonl(directory / f"{phase}_outcomes.jsonl")
        cases = make_cases(manifest, phase)
        validate_records(rows, cases, digest(manifest_path))
        if len(rows) != len(cases):
            raise ValueError("report requires complete development and test data")
        phases.append(sorted(rows, key=lambda r: r["case_id"]))
    development, test = phases
    if {r["question_id"] for r in development} & {r["question_id"] for r in test}:
        raise ValueError("development/test question leakage")
    return manifest, development, test, OutcomeRouter.load(model_path)


def report_markdown(report: dict) -> str:
    lines = [
        "# Outcome routing and generalization results", "",
        "Frozen outcome-trained policy. One provider seed; 60 held-out HotpotQA questions.", "",
        "Costs and latency below cover selected incremental repair actions only.", "",
    ]
    for name in ("seen", "unseen", "natural"):
        group = report["groups"][name]
        lines.extend([
            f"## {name.capitalize()}", "",
            f"{group['cases']} cases; {group['initially_wrong']} initially wrong; {group['initially_correct']} initially correct.", "",
            "| Policy | EM | Token F1 | Recovery | Damage | Calls/case | Tokens/case | Cost/case | Mean latency |",
            "|:--|--:|--:|--:|--:|--:|--:|--:|--:|",
        ])
        for policy, values in group["policies"].items():
            usage = values["incremental"]
            rates = [values[key]["estimate"] for key in ("exact_match", "token_f1", "recovery", "damage")]
            formatted = ["—" if value is None else f"{value:.1%}" for value in rates]
            lines.append(f"| {policy} | {' | '.join(formatted)} | {usage['calls_per_case']:.2f} | "
                         f"{usage['tokens_per_case']:.0f} | ${usage['cost_usd_per_case']['estimate']:.6f} | "
                         f"{usage['latency_seconds_mean']:.3f}s |")
        lines.extend(["", "Paired EM differences (learned minus comparator), percentage points:", ""])
        for comparison, metrics in group["comparisons"].items():
            effect = metrics["exact_match"]
            low, high = effect["ci"]
            lines.append(f"- {comparison}: {100 * effect['estimate']:+.1f} [{100 * low:.1f}, {100 * high:.1f}]")
        lines.extend(["", "Learned-policy rates with 95% intervals:", "",
                      "| Metric | Estimate | 95% CI | Eligible cases |",
                      "|:--|--:|:--|--:|"])
        for metric in ("exact_match", "token_f1", "recovery", "damage"):
            value = group["policies"]["learned"][metric]
            estimate = "—" if value["estimate"] is None else f"{value['estimate']:.1%}"
            interval = "—" if value["ci"] is None else f"[{value['ci'][0]:.1%}, {value['ci'][1]:.1%}]"
            lines.append(f"| {metric} | {estimate} | {interval} | {value['eligible_count']} |")
        lines.append("")
    lines.extend([
        "## Interpretation limits", "",
        "Intervals are paired, question-clustered 95% percentile intervals (10,000 resamples). "
        "They are pointwise, not multiplicity-adjusted. Natural cases are unmodified benchmark "
        "outputs, not production data. All unseen variants come from the same HotpotQA corpus. "
        "Exact match and token F1 are lexical metrics; neither establishes evidence entailment. "
        "Latency reflects three-worker API collection, includes local rate-limit waits, "
        "and excludes router CPU overhead. The JSON separately records throttle time "
        "and measured wall time minus local throttle waits; these are not deployment benchmarks.", "",
        "The JSON report includes per-family results, natural-error-only results, full intervals, "
        "action distributions, p95 latency, stage timings, and the difference between "
        "counterfactual collection cost and selected-policy cost.", "",
    ])
    return "\n".join(lines)


def create_report(directory: Path) -> dict:
    manifest, development, test, policy = read_frozen_study(directory)
    choices = {row["case_id"]: policy.decide(question=row["question"], initial=row["initial"]) for row in test}
    groups = {
        "seen": [r for r in test if r["family"] in SEEN_FAMILIES],
        "unseen": [r for r in test if r["family"] in UNSEEN_FAMILIES],
        "natural": [r for r in test if r["family"] == "natural"],
    }
    groups["natural_errors"] = [r for r in groups["natural"] if not score(r["initial"]["answer"], r["gold"])["exact_match"]]
    for family in (*SEEN_FAMILIES, *UNSEEN_FAMILIES):
        groups[family] = [r for r in test if r["family"] == family]
    observations = development + test
    telemetry = [r["outcomes"][a]["telemetry"] for r in observations for a in ACTIONS]
    setup = [r["setup_telemetry"] for r in observations]
    ledger = load_jsonl(directory / "budget.jsonl")
    result = {
        "complete": True, "manifest_sha256": digest(directory / "manifest.json"),
        "model_sha256": digest(directory / "model.json"), "provider_seed": 0,
        "development_cases": len(development), "test_cases": len(test),
        "statistics": {"method": "paired cluster percentile bootstrap by question ID",
                       "confidence": .95, "resamples": manifest["resamples"], "seed": 17,
                       "multiplicity_adjusted": False},
        "selected_policy": {"ridge": policy.ridge, "min_gain": policy.min_gain, "cost_penalty": policy.cost_penalty},
        "groups": {name: evaluate_group(rows, choices, resamples=manifest["resamples"]) for name, rows in groups.items()},
        "collection": {
            "completed_case_calls": sum(t["llm_calls"] for t in telemetry + setup),
            "completed_case_tokens": sum(t["total_tokens"] for t in telemetry + setup),
            "completed_case_cost_usd": sum(t["estimated_cost_usd"] for t in telemetry + setup),
            "new_initial_state_cost_usd": sum(t["estimated_cost_usd"] for t in setup),
            "reserved_upper_usd": sum(entry["reserved_usd"] for entry in ledger),
            "request_reservations": len(ledger), "reservation_cap_usd": ledger[0]["limit_usd"],
            "note": "Reservation ledger also retains failed or discarded requests; completed-case cost may omit those.",
        },
        "evaluation_note": (
            "Full-information paired offline evaluation from observed actions, not an online deployment. "
            "Selected-policy costs/timings are incremental, excluding historical initial passes and router CPU. "
            "Naturally occurring errors are EM failures of unmodified benchmark baselines. "
            "One dataset and provider seed; no general production or cross-dataset claim."
        ),
    }
    (directory / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (directory / "report.md").write_text(report_markdown(result))
    with (directory / "test_policy_choices.jsonl").open("w") as handle:
        for row in test:
            handle.write(json.dumps({"case_id": row["case_id"], "action": choices[row["case_id"]],
                                     "predicted_utilities": policy.utilities(row["features"]).tolist()}) + "\n")
    print(f"Wrote {directory / 'report.md'}")
    return result
