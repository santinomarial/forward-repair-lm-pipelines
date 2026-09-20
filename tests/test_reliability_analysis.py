import copy
import json

import numpy as np
import pytest

from outcome_routing import ACTIONS, runtime_features
from reliability_analysis import ClusterBootstrap, create_report, evaluate_group
from reliability_study import digest, fit, make_cases, question_splits, score, write_new, zero_telemetry


def case_row(case, initial_answer="wrong", final_answer="France"):
    initial = {"query": "Paris France", "answer": initial_answer, "context": "Paris is in France.",
               "docs": [{"id": "Paris", "title": "Paris", "text": "Paris is in France.", "score": 1}]}
    outcomes = {}
    for action in ACTIONS:
        answer = initial_answer if action == "accept" else final_answer
        telemetry = zero_telemetry()
        if action != "accept":
            telemetry.update(llm_calls=1, prompt_tokens=9, completion_tokens=1, total_tokens=10,
                             estimated_cost_usd=.001, priced_calls=1, wall_clock_seconds=1)
            telemetry["latency_seconds"]["answer_generation"] = 1
        outcomes[action] = {"answer": answer, "metrics": score(answer, "France"), "telemetry": telemetry}
    return {**case, "question": "Where is Paris?", "gold": "France", "initial": initial,
            "features": runtime_features("Where is Paris?", initial), "outcomes": outcomes,
            "setup_telemetry": zero_telemetry()}


def test_bootstrap_keeps_correlated_variants_together():
    bootstrap = ClusterBootstrap(["same_question", "same_question"], resamples=100)
    result = bootstrap.ratio([0, 1])
    assert result["estimate"] == .5
    assert result["ci"] == [.5, .5]
    assert bootstrap.ratio([0, 0], [0, 0])["estimate"] is None
    with pytest.raises(ValueError):
        bootstrap.ratio([0])
    with pytest.raises(ValueError):
        ClusterBootstrap([])


def test_group_reports_damage_recovery_cost_and_paired_effects():
    rows = []
    for i, (initial, final) in enumerate([("wrong", "France"), ("wrong", "wrong"), ("France", "wrong"), ("France", "France")]):
        rows.append(case_row({"case_id": str(i), "question_id": str(i // 2)}, initial, final))
    result = evaluate_group(rows, {str(i): "regenerate" for i in range(4)}, resamples=100)
    learned = result["policies"]["learned"]
    assert learned["exact_match"]["estimate"] == .5
    assert learned["recovery"]["estimate"] == .5
    assert learned["damage"]["estimate"] == .5
    assert learned["incremental"]["cost_usd_total"] == .004
    assert learned["incremental"]["latency_seconds_p95"] == 1
    assert learned["incremental"]["throttle_seconds_mean"] == 0
    rows[0]["outcomes"]["regenerate"]["telemetry"]["throttle_seconds"] = .4
    paced = evaluate_group(rows, {str(i): "regenerate" for i in range(4)}, resamples=100)
    assert paced["policies"]["learned"]["incremental"]["throttle_seconds_mean"] == .1
    assert paced["policies"]["learned"]["incremental"]["unthrottled_seconds_mean"] == .9
    assert result["questions"] == 2
    assert result["comparisons"]["learned_minus_accept"]["exact_match"]["estimate"] == 0
    assert evaluate_group([], {})["cases"] == 0


def make_artifacts(directory):
    manifest = {"splits": question_splits([str(i) for i in range(5)]), "cost_penalty": 100.0,
                "resamples": 20}
    manifest_path = directory / "manifest.json"
    write_new(manifest_path, manifest)
    for phase in ("development", "test"):
        rows = [case_row(case) for case in make_cases(manifest, phase)]
        for row in rows:
            row["manifest_sha256"] = digest(manifest_path)
        (directory / f"{phase}_outcomes.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    fit(directory)
    write_new(directory / "test_model_lock.json", {
        "model_sha256": digest(directory / "model.json"), "manifest_sha256": digest(manifest_path),
    })
    (directory / "budget.jsonl").write_text(json.dumps({"limit_usd": 3, "reserved_usd": 1}) + "\n")


def test_offline_report_enforces_freeze_and_completeness(tmp_path):
    make_artifacts(tmp_path)
    report = create_report(tmp_path)
    assert report["complete"]
    assert report["test_cases"] == 5
    assert report["groups"]["natural"]["policies"]["learned"]["damage"]["estimate"] is None
    assert (tmp_path / "report.md").exists()
    assert report == create_report(tmp_path)
    with pytest.raises(ValueError, match="cannot fit"):
        fit(tmp_path)
    test_path = tmp_path / "test_outcomes.jsonl"
    test_path.write_text("\n".join(test_path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="complete"):
        create_report(tmp_path)


def test_report_rejects_modified_model(tmp_path):
    make_artifacts(tmp_path)
    path = tmp_path / "model.json"
    model = json.loads(path.read_text())
    model["policy"]["min_gain"] = .123
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="changed"):
        create_report(tmp_path)


def test_clusters_and_pairing_are_reproducible():
    ids = ["a", "a", "b", "b", "c"]
    first = ClusterBootstrap(ids, resamples=200)
    second = ClusterBootstrap(ids, resamples=200)
    assert np.array_equal(first.indices, second.indices)
    assert first.ratio([0, 0, 0, 0, 0])["ci"] == [0, 0]
