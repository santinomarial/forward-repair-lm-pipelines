import copy
import hashlib
import json

import pytest

import case_explorer as explorer
from outcome_routing import ACTIONS, OUTCOME_FEATURES, OutcomeRouter, runtime_features
from reliability_study import context_for, digest, score, zero_telemetry


@pytest.fixture
def snapshot_fixture(tmp_path, monkeypatch):
    corpus = [{"id": "paris", "title": "Paris", "text": "Paris is in France."}]
    context = context_for(corpus)
    row = {"case_id": "q:natural", "question_id": "q", "family": "natural", "question": "Where is Paris?",
           "gold": "France", "initial": {"query": "Paris", "answer": "wrong", "docs": corpus, "context": context},
           "outcomes": {}}
    for action in ACTIONS:
        answer = "wrong" if action == "accept" else "France"
        row["outcomes"][action] = {"answer": answer, "query": "Paris", "metrics": score(answer, "France"),
                                    "doc_ids": ["paris"], "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                                    "telemetry": zero_telemetry()}
    row["features"] = runtime_features(row["question"], row["initial"])
    coefs = [[0.0] * len(ACTIONS) for _ in range(len(OUTCOME_FEATURES))] + [[0, .5, .2, .1]]
    policy = OutcomeRouter([0.0] * len(OUTCOME_FEATURES), [1.0] * len(OUTCOME_FEATURES), coefs, 100, 1, .1)
    examples = [{"id": "q", "question": row["question"], "answer": "France", "support_doc_ids": ["paris"]}]
    for filename, rows in (("corpus.jsonl", corpus), ("examples.jsonl", examples), ("test_outcomes.jsonl", [row])):
        (tmp_path / filename).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for filename in ("manifest.json", "model.json"):
        (tmp_path / filename).write_text("{}")
    manifest = {"sources": {"corpus": {"sha256": digest(tmp_path / "corpus.jsonl")}}}
    monkeypatch.setattr(explorer, "read_frozen_study", lambda path: (manifest, [], [row], policy))
    return tmp_path, row, corpus, manifest


def load_fixture(fixture):
    directory, *_ = fixture
    return explorer.load_snapshot(directory, directory / "corpus.jsonl", directory / "examples.jsonl")


def test_load_replay_recomputes_policy_and_exports_exact_evidence(snapshot_fixture):
    snapshot = load_fixture(snapshot_fixture)
    case = snapshot.cases[0]
    assert case["action"] == "rewrite_query"
    assert case["transition"] == "Recovered"
    payload = json.loads(explorer.case_export(snapshot, case))
    assert payload["action_documents"]["expand_context"][0]["id"] == "paris"
    assert "test_outcomes.jsonl" in payload["provenance"]
    assert len(case["features"]) == len(OUTCOME_FEATURES)


@pytest.mark.parametrize("mutation, message", [
    ("context", "context hash"), ("order", "document order"), ("missing", "missing from"),
    ("features", "features"), ("accept", "preserve"), ("metrics", "metrics"),
    ("annotations", "annotations"), ("corpus", "corpus differs"), ("duplicate", "duplicate"),
    ("support", "support missing"),
])
def test_replay_refuses_inconsistent_artifacts(snapshot_fixture, mutation, message):
    path, row, corpus, manifest = snapshot_fixture
    if mutation == "context":
        row["outcomes"]["regenerate"]["context_sha256"] = "changed"
    elif mutation == "order":
        row["outcomes"]["accept"]["doc_ids"] = []
    elif mutation == "missing":
        row["outcomes"]["rewrite_query"]["doc_ids"] = ["absent"]
    elif mutation == "features":
        row["features"][0] += 1
    elif mutation == "accept":
        row["outcomes"]["accept"]["answer"] = "changed"
    elif mutation == "metrics":
        row["outcomes"]["regenerate"]["metrics"]["exact_match"] = 0
    elif mutation in ("annotations", "support"):
        example = json.loads((path / "examples.jsonl").read_text())
        example["answer" if mutation == "annotations" else "support_doc_ids"] = "changed" if mutation == "annotations" else ["absent"]
        (path / "examples.jsonl").write_text(json.dumps(example))
    elif mutation == "corpus":
        manifest["sources"]["corpus"]["sha256"] = "changed"
    else:
        (path / "corpus.jsonl").write_text("\n".join(json.dumps(corpus[0]) for _ in range(2)))
        manifest["sources"]["corpus"]["sha256"] = digest(path / "corpus.jsonl")
    with pytest.raises(ValueError, match=message):
        load_fixture(snapshot_fixture)


@pytest.mark.parametrize("answer,gold,expected", [
    ("It was France.", "France", True), ("France", "France", False),
    ("Anne", "Ann", False), ("I said no, then yes", "yes", False), ("nothing", "", False),
    ("Not France, actually Germany", "France", True),
])
def test_format_signal_is_only_a_review_candidate(answer, gold, expected):
    assert explorer.format_candidate(answer, gold) is expected


def test_filters_hindsight_and_unknown_actions(snapshot_fixture):
    snapshot = load_fixture(snapshot_fixture)
    row = copy.deepcopy(snapshot_fixture[1])
    case = explorer.annotate_case(row, "accept", ["unretrieved"])
    assert case["missed_recovery"]
    assert case["missing_support_before"] == ["unretrieved"]
    assert explorer.filter_cases([case], view="Missed recovery", search="PARIS", family="natural") == [case]
    assert explorer.filter_cases([case], search="nonexistent") == []
    assert explorer.filter_cases([case], family="vague_query") == []
    for view, flag in (("Kept correct", "kept_correct"), ("Format candidates", "format_candidate")):
        assert explorer.filter_cases([{**case, flag: True}], view=view)
    with pytest.raises(ValueError):
        explorer.filter_cases([], view="unknown")
    with pytest.raises(ValueError):
        explorer.evidence_for(row, "unknown", snapshot.corpus)


def test_published_snapshot_loads_without_any_provider_calls(monkeypatch):
    import dspy
    monkeypatch.setattr(dspy.LM, "forward", lambda *a, **k: pytest.fail("offline replay called provider"))
    snapshot = explorer.load_snapshot()
    assert len(snapshot.cases) == 300
    assert len(explorer.filter_cases(snapshot.cases, view="Harmed")) == 4
    assert len(explorer.filter_cases(snapshot.cases, view="Recovered")) == 57
