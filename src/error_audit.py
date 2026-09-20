"""Generate a deterministic offline audit and a pending human-review queue."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

from case_explorer import Snapshot, load_snapshot
from config import ERROR_AUDIT_DIR, RELIABILITY_STUDY_DIR


REVIEW_LABELS = ("format_only", "missing_evidence", "reasoning_error", "unsupported_claim",
                 "reference_ambiguity", "no_error", "uncertain")


def review_sample(cases: list[dict], size: int = 20) -> list[dict]:
    if size <= 0:
        raise ValueError("sample size must be positive")
    ordered = sorted(cases, key=lambda c: hashlib.sha256(c["case_id"].encode()).hexdigest())
    # Include damage first, then round-robin over family/transition strata.
    selected = [c for c in ordered if c["transition"] == "Harmed"][:size]
    chosen = {c["case_id"] for c in selected}
    groups = defaultdict(list)
    for case in ordered:
        if case["case_id"] not in chosen:
            groups[(case["family"], case["transition"])].append(case)
    while len(selected) < min(size, len(cases)):
        for key in sorted(groups):
            if groups[key] and len(selected) < size:
                selected.append(groups[key].pop(0))
    return selected


def review_template(snapshot: Snapshot, case: dict) -> dict:
    row = snapshot.rows[case["case_id"]]
    return {"case_id": case["case_id"], "family": case["family"], "question": row["question"],
            "reference_answer": row["gold"], "initial_answer": row["initial"]["answer"],
            "selected_answer": row["outcomes"][case["action"]]["answer"],
            "action": case["action"], "transition": case["transition"], "automatic_flags": case["flags"],
            "status": "pending", "reviewer": "", "before_labels": [], "after_labels": [],
            "evidence_doc_ids": [], "rationale": "", "provenance": snapshot.provenance}


def completed_review(template: dict, *, reviewer: str, before_labels: list[str], after_labels: list[str],
                     evidence_doc_ids: list[str], rationale: str, allowed_doc_ids: set[str]) -> dict:
    """Only explicit reviewer input becomes a reviewed annotation; not an auto-label."""
    if not reviewer.strip() or not rationale.strip() or not before_labels or not after_labels:
        raise ValueError("provide reviewer, both label sets, and an evidence-based rationale")
    if not set(before_labels + after_labels).issubset(REVIEW_LABELS):
        raise ValueError("unknown review label")
    if not set(evidence_doc_ids).issubset(allowed_doc_ids):
        raise ValueError("evidence IDs must belong to this case")
    if not evidence_doc_ids and "uncertain" not in before_labels + after_labels:
        raise ValueError("cite evidence or explicitly mark uncertainty")
    return {**template, "status": "reviewed", "reviewer": reviewer.strip(),
            "before_labels": before_labels, "after_labels": after_labels,
            "evidence_doc_ids": evidence_doc_ids, "rationale": rationale.strip()}


def build_audit(snapshot: Snapshot, sample_size: int = 20) -> tuple[dict, list[dict]]:
    cases = snapshot.cases
    queue = [review_template(snapshot, c) for c in review_sample(cases, sample_size)]
    groups = {}
    for family in ["all", *sorted({c["family"] for c in cases})]:
        rows = cases if family == "all" else [c for c in cases if c["family"] == family]
        groups[family] = {
            "cases": len(rows), "transitions": dict(Counter(c["transition"] for c in rows)),
            "automatic_flags": dict(Counter(flag for c in rows for flag in c["flags"])),
            "kept_correct": sum(c["kept_correct"] for c in rows),
            "missed_recovery": sum(c["missed_recovery"] for c in rows),
            "recovery_with_gold_already_present": sum(c["recovery_with_gold_already_present"] for c in rows),
        }
    return {"provenance": snapshot.provenance, "groups": groups,
            "review_sample": {"size": len(queue), "reviewed": 0,
                              "selection": "all damage first (up to limit), then SHA256-ordered family/transition round-robin"},
            "note": "Post-hoc descriptive signals, not human judgments or estimates of semantic error prevalence. "
                    "Support annotations are audit-only, never router inputs. Review sample is intentionally nonrepresentative."}, queue


def audit_markdown(summary: dict) -> str:
    lines = ["# Saved-case error audit", "", summary["note"], "",
             "| Family | Cases | Recovered (EM) | Harmed (EM) | Correct answer kept without repair | Missed recovery |",
             "|:--|--:|--:|--:|--:|--:|"]
    for family, group in summary["groups"].items():
        counts = group["transitions"]
        lines.append(f"| {family} | {group['cases']} | {counts.get('Recovered', 0)} | {counts.get('Harmed', 0)} | "
                     f"{group['kept_correct']} | {group['missed_recovery']} |")
    lines.extend(["", "Missed recovery means another saved action achieved EM=1 when the selected action did not. "
                  "It is a hindsight diagnostic, not an available runtime oracle.", "", "## Automatic signals", ""])
    for flag, count in sorted(summary["groups"]["all"]["automatic_flags"].items()):
        lines.append(f"- {flag}: {count}")
    group = summary["groups"]["all"]
    lines.extend(["", f"Of {group['transitions'].get('Recovered', 0)} EM recoveries, "
                  f"{group['recovery_with_gold_already_present']} already contained the whole normalized gold phrase before repair "
                  "(excluding yes/no). This is a formatting-review candidate, not proof the original answer was correct.", "",
                  "## Human review", "", f"The queue contains {summary['review_sample']['size']} cases; **0 are human-reviewed**. "
                  "All labels start empty. Damage cases are prioritized, then family/transition strata are sampled deterministically. "
                  "This sample must not be used to estimate population error rates.", "",
                  "Use the explorer to inspect before/after evidence and export explicit reviewer annotations. "
                  "Missing annotated support does not prove insufficient evidence; complete support does not prove correct reasoning. "
                  "Unsupported claims and reasoning errors require evidence review, not substring rules.", ""])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=RELIABILITY_STUDY_DIR)
    parser.add_argument("--output", type=Path, default=ERROR_AUDIT_DIR)
    parser.add_argument("--sample-size", type=int, default=20)
    args = parser.parse_args()
    snapshot = load_snapshot(args.study)
    summary, queue = build_audit(snapshot, args.sample_size)
    args.output.mkdir(parents=True, exist_ok=True)
    # These are derived files only. Never overwrite reviewer annotations.
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.output / "review_queue.jsonl").write_text("".join(json.dumps(r) + "\n" for r in queue))
    (args.output / "report.md").write_text(audit_markdown(summary))
    print(f"Wrote offline audit to {args.output}; no model calls; human review pending.")


if __name__ == "__main__":
    main()
