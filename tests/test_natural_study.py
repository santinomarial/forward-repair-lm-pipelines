import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import dspy
import numpy as np
import pytest

import natural_analysis as analysis
import natural_study as study
from data_loader import load_jsonl
from natural_routing import NATURAL_ACTIONS, NaturalRouter, select_natural_router, utility
from outcome_routing import ACTIONS, OUTCOME_FEATURES, OutcomeRouter
from reliability_study import digest, write_new


@pytest.fixture
def mocked_models(monkeypatch):
    class LM:
        def __init__(self, *args, **kwargs):
            self.history = []
            self.throttle_seconds = 0.0

    active = []

    def new_lm(*args, **kwargs):
        lm = LM()
        active.append(lm)
        return lm

    def record():
        lm = dspy.settings.lm
        lm.history.append({"usage": {"prompt_tokens": 40, "completion_tokens": 2}, "cost": .00001})

    class Query:
        def __call__(self, **kwargs):
            assert set(kwargs) <= {"question", "bad_query"}
            record()
            return SimpleNamespace(query="France" if "bad_query" in kwargs else "Paris")

    class Answer:
        def __call__(self, **kwargs):
            assert set(kwargs) == {"question", "context"}
            record()
            return SimpleNamespace(answer="France" if len(dspy.settings.lm.history) > 2 else "UNKNOWN")

    monkeypatch.setattr(study, "BudgetedLM", new_lm)
    monkeypatch.setattr(study, "QueryGenerator", lambda mode: Query())
    monkeypatch.setattr(study, "AnswerGenerator", Answer)
    monkeypatch.setattr(study, "OPENAI_API_KEY", "not-a-real-key")
    return new_lm


@pytest.fixture
def small_study(tmp_path):
    directory = tmp_path / "natural"
    directory.mkdir()
    examples = [{"id": str(i), "question": f"Where is Paris {i}?", "answer": "France",
                 "support_doc_ids": ["0", "1"],
                 "split": "train" if i < 4 else "validation" if i < 6 else "test"}
                for i in range(9)]
    corpus = [{"id": str(i), "title": f"France Paris {i}", "text": f"Paris is in France {i}."} for i in range(12)]
    study.write_rows_new(directory / "examples.jsonl", examples)
    study.write_rows_new(directory / "corpus.jsonl", corpus)
    n = len(OUTCOME_FEATURES)
    OutcomeRouter([0.] * n, [1.] * n, np.zeros((n+1, 4)).tolist(), 100., 1.).save(
        directory / "synthetic_model.json", {})
    manifest = {"code_sha256": study.code_hashes(), "dspy_version": study.version("dspy"),
                "budget_cap_usd": 5., "sources": {name: digest(directory / name) for name in (
                    "examples.jsonl", "corpus.jsonl", "synthetic_model.json")},
                "splits": {s: [r["id"] for r in examples if r["split"] == s] for s in ("train", "validation", "test")},
                "pilot_ids": ["0", "1"], "resamples": 100,
                "primary_comparison": "natural_damage_augmented_minus_accept"}
    write_new(directory / "manifest.json", manifest)
    # Fake LM bypasses reservation; a valid synthetic ledger supports report tests.
    study.write_rows_new(directory / "budget.jsonl", [{"reserved_usd": .01, "limit_usd": 5}])
    return directory


def test_preserving_merge_never_drops_original_and_deduplicates():
    docs = [{"id": str(i)} for i in range(15)]
    assert study.preserve_documents(docs[:5], docs[3:13]) == docs[:10]
    assert study.preserve_documents(docs[:5], docs[:5]) == docs[:5]
    with pytest.raises(ValueError, match="preserve"):
        study.preserve_documents(docs, [], 10)


def test_actual_calls_and_selected_policy_costs_are_distinct(small_study, mocked_models):
    corpus = load_jsonl(small_study / "corpus.jsonl")
    lm = mocked_models()
    with dspy.context(lm=lm):
        row = study.run_question(load_jsonl(small_study / "examples.jsonl")[0], study.BM25Retriever(corpus), lm)
    assert row["collection_telemetry"]["llm_calls"] == 8
    assert row["baseline_telemetry"]["llm_calls"] == 2
    assert row["outcomes"]["accept"]["telemetry"]["llm_calls"] == 0
    for action in ("rewrite_query", "rewrite_query_10", "preserve_evidence"):
        assert row["outcomes"][action]["telemetry"]["llm_calls"] == 2
        assert row["outcomes"][action]["query"] == "France"
    assert sum(o["telemetry"]["llm_calls"] for o in row["outcomes"].values()) == 8
    analysis.verify_evidence(small_study, [row])
    changed = copy.deepcopy(row)
    changed["outcomes"]["preserve_evidence"]["context_sha256"] = "tampered"
    with pytest.raises(ValueError, match="hash"):
        analysis.verify_evidence(small_study, [changed])


def test_full_offline_lifecycle_and_frozen_test_guards(small_study, mocked_models):
    directory = small_study
    study.collect(directory, phase="pilot", workers=1, limit=1)
    with pytest.raises(ValueError, match="finish pilot"):
        analysis.pilot_report(directory)
    with pytest.raises(ValueError, match="finish development"):
        study.fit(directory)
    with pytest.raises(FileExistsError, match="resume"):
        study.collect(directory, phase="pilot", workers=1)
    study.collect(directory, phase="pilot", workers=1, resume=True)
    pilot = analysis.pilot_report(directory)
    assert pilot["questions"] == 2
    assert not pilot["sample_size_changed"]
    study.collect(directory, phase="development", workers=1, resume=True)
    models = study.fit(directory)
    assert set(models["policies"]) == {"natural_original", "natural_augmented", "natural_damage_augmented"}
    study.collect(directory, phase="test", workers=1)
    study.collect(directory, phase="test", workers=1, resume=True)
    with pytest.raises(ValueError, match="after test"):
        study.fit(directory)
    with pytest.raises(ValueError, match="forbidden"):
        study.collect(directory, phase="development", workers=1, resume=True)
    with pytest.raises(ValueError, match="precede"):
        analysis.pilot_report(directory)
    report = analysis.create_report(directory)
    assert report["test_questions"] == 3
    assert report["collection"]["completed_calls"] == 72
    assert report["policies"]["preserve_evidence"]["recovered_count"] == 3
    assert report["policies"]["accept"]["damage"]["estimate"] is None
    review_path = directory / "my_human_reviews.jsonl"
    review_path.write_text("untouched")
    assert analysis.create_report(directory) == report
    assert review_path.read_text() == "untouched"
    lock = json.loads((directory / "test_model_lock.json").read_text())
    lock["models_sha256"] = "tampered"
    (directory / "test_model_lock.json").write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="changed"):
        analysis.create_report(directory)
    with pytest.raises(ValueError, match="changed"):
        study.collect(directory, phase="test", workers=1, resume=True)


def test_checkpoint_validation_and_failure_resume(small_study, mocked_models, monkeypatch):
    original = study.run_question
    calls = []

    def failing(example, *args):
        calls.append(example["id"])
        if len(calls) == 2:
            raise RuntimeError("provider failed")
        return original(example, *args)

    monkeypatch.setattr(study, "run_question", failing)
    with pytest.raises(RuntimeError, match="preserved"):
        study.collect(small_study, phase="development", workers=1)
    assert len(load_jsonl(small_study / "development_outcomes.jsonl")) == 1
    monkeypatch.setattr(study, "run_question", original)
    study.collect(small_study, phase="development", workers=1, resume=True)
    rows = load_jsonl(small_study / "development_outcomes.jsonl")
    for mutate in (lambda r: r.update(split="test"), lambda r: r.update(gold="tampered"),
                   lambda r: r.update(features=[0]*11), lambda r: r["initial"]["metrics"].update(exact_match=1)):
        changed = copy.deepcopy(rows)
        mutate(changed[0])
        with pytest.raises(ValueError):
            study.validate_rows(small_study, changed, ("train", "validation"))
    with pytest.raises(ValueError, match="duplicate"):
        study.validate_rows(small_study, rows + rows, ("train", "validation"))


def test_input_integrity_and_cli_guards(small_study, mocked_models, monkeypatch):
    with pytest.raises(ValueError, match="phase"):
        study.collect(small_study, phase="invalid")
    with pytest.raises(ValueError, match="positive"):
        study.collect(small_study, phase="pilot", limit=0)
    monkeypatch.setattr(study, "OPENAI_API_KEY", None)
    with pytest.raises(RuntimeError, match="missing"):
        study.collect(small_study, phase="pilot")
    manifest = json.loads((small_study / "manifest.json").read_text())
    for key, value, message in (("code_sha256", {}, "code changed"), ("budget_cap_usd", 6, "budget changed"),
                                ("sources", {"examples.jsonl": "bad"}, "input changed")):
        (small_study / "manifest.json").write_text(json.dumps({**manifest, key: value}))
        with pytest.raises(ValueError, match=message):
            study.verify_inputs(small_study)


def test_router_whitelist_damage_target_and_split_guards(small_study, mocked_models):
    study.collect(small_study, phase="development", workers=1)
    rows = load_jsonl(small_study / "development_outcomes.jsonl")
    train, validation = rows[:4], rows[4:]
    payload, selection = select_natural_router(train, validation, actions=NATURAL_ACTIONS, damage_penalty=1.)
    router = NaturalRouter(**payload)
    modified = copy.deepcopy(validation[0])
    modified.update(gold="FAKE", family="FAKE", question_id="FAKE", outcomes={})
    assert router.decide(modified) == router.decide(validation[0])
    assert len(selection["candidates"]) == 12
    harmful = copy.deepcopy(train[0])
    harmful["initial"]["metrics"]["exact_match"] = 1
    harmful["outcomes"]["regenerate"]["metrics"]["exact_match"] = 0
    assert utility(harmful, "regenerate", 1) == pytest.approx(utility(harmful, "regenerate", 0)-1)
    with pytest.raises(ValueError, match="features"):
        router.utilities([float("nan")] * len(OUTCOME_FEATURES))
    for t, v, a, p in ((train, validation, ("accept",), 0), (train, validation, ACTIONS, 2),
                        ([], validation, ACTIONS, 0), (train + train, validation, ACTIONS, 0),
                        ([{**r, "split": "test"} for r in train], validation, ACTIONS, 0),
                        (train, [{**validation[0], "question_id": train[0]["question_id"]}], ACTIONS, 0)):
        with pytest.raises(ValueError):
            select_natural_router(t, v, actions=a, damage_penalty=p)


def test_unpriced_and_oversized_generation_fail_closed(small_study, mocked_models, monkeypatch):
    lm = mocked_models()
    def unpriced():
        lm.history.append({"usage": {"prompt_tokens": 1}})
    with pytest.raises(RuntimeError, match="unpriced"):
        study.measured(lm, "query_generation", unpriced)
    class Retriever:
        def retrieve(self, query, top_k):
            return [{"id": "huge", "title": "Big", "text": "word "*4000}]
    with dspy.context(lm=lm), pytest.raises(ValueError, match="ceiling"):
        study.run_question(load_jsonl(small_study / "examples.jsonl")[0], Retriever(), lm)


def test_prepare_is_disjoint_bounded_and_immutable(tmp_path, monkeypatch):
    import datasets
    class Dataset(list):
        _fingerprint = "test-data"
    dataset = Dataset([{
        "id": f"fresh-{i}", "question": f"Unique question {i}?", "answer": "France",
        "context": {"title": ["France", "Paris"], "sentences": [["France "*400], ["Paris is in France."]]},
        "supporting_facts": {"title": ["France", "Paris"]},
    } for i in range(610)])
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **k: dataset)
    directory = tmp_path / "prepared"
    study.prepare(directory)
    manifest = study.verify_inputs(directory)
    assert [len(manifest["splits"][s]) for s in ("train", "validation", "test")] == [240, 60, 300]
    assert set(manifest["pilot_ids"]) <= set(manifest["splits"]["train"])
    assert any(r["truncated"] for r in load_jsonl(directory / "corpus.jsonl"))
    with pytest.raises(FileExistsError, match="already"):
        study.prepare(directory)


def test_cli_dispatch(small_study, monkeypatch):
    import sys
    for command, target, module in (("prepare", "prepare", study), ("collect", "collect", study),
                                    ("fit", "fit", study), ("report", "create_report", analysis),
                                    ("pilot-report", "pilot_report", analysis)):
        calls = []
        monkeypatch.setattr(module, target, lambda directory, **kwargs: calls.append(directory))
        monkeypatch.setattr(sys, "argv", ["natural_study", command, "--directory", str(small_study)])
        study.main()
        assert calls == [small_study]
