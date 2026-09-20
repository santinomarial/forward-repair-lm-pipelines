import copy
import json
import sys
from types import SimpleNamespace

import pytest

import reliability_study as study
from outcome_routing import ACTIONS, SEEN_FAMILIES


@pytest.fixture
def corpus():
    return [{"id": str(i), "title": f"Doc {i}", "text": "Paris is in France.", "score": float(10-i)} for i in range(10)]


def test_grouped_splits_and_test_only_families():
    splits = study.question_splits([str(i) for i in range(300)])
    assert {k: len(v) for k, v in splits.items()} == {"train": 180, "validation": 60, "test": 60}
    assert not (set(splits["train"]) & set(splits["test"]))
    assert not (set(splits["validation"]) & set(splits["test"]))
    manifest = {"splits": splits}
    dev, test = study.make_cases(manifest, "development"), study.make_cases(manifest, "test")
    assert len(dev) == 480
    assert len(test) == 300
    assert {r["family"] for r in dev} == set(SEEN_FAMILIES)
    assert {r["question_id"] for r in dev}.isdisjoint(r["question_id"] for r in test)


def test_substitution_is_deterministic_and_uses_no_labels(corpus):
    first = study.substitute_query("Paris France geography", corpus[:5], corpus, "q")
    assert first == study.substitute_query("Paris France geography", corpus[:5], corpus, "q")
    assert first[0] != "Paris France geography"
    assert first[1]["removed_term"] == "geography"
    with pytest.raises(ValueError):
        study.substitute_query("123", corpus, corpus, "q")


@pytest.mark.parametrize("family", ["vague_query", "ignore_context", "natural", "query_term_substitution", "retrieval_rank_dropout"])
def test_case_actions_are_fixed_and_labels_never_reach_modules(monkeypatch, corpus, family):
    lm = SimpleNamespace(history=[])
    calls = []

    class Answerer:
        def __call__(self, **kwargs):
            assert set(kwargs) == {"question", "context"}
            calls.append(kwargs)
            lm.history.append({"usage": {"prompt_tokens": 10, "completion_tokens": 1}, "cost": .001})
            return SimpleNamespace(answer="France")

    class Rewriter:
        def __call__(self, **kwargs):
            assert set(kwargs) == {"question", "bad_query"}
            lm.history.append({"usage": {"prompt_tokens": 5, "completion_tokens": 1}, "cost": .001})
            return SimpleNamespace(query="Paris France")

    class Retriever:
        def retrieve(self, query, top_k):
            return corpus[:top_k]

    monkeypatch.setattr(study, "AnswerGenerator", Answerer)
    monkeypatch.setattr(study, "QueryGenerator", lambda mode: Rewriter())
    source = {"question": "Where is Paris?", "gold": "France",
              "run": {"query": "Paris geography", "docs": corpus[:5],
                      "context": study.context_for(corpus[:5]), "answer": "wrong"}}
    before = copy.deepcopy(source)
    case = {"case_id": "q:" + family, "question_id": "q", "split": "test", "family": family}
    result = study.run_case(case, source, corpus, Retriever(), lm)
    assert source == before
    assert set(result["outcomes"]) == set(ACTIONS)
    assert result["outcomes"]["accept"]["telemetry"]["llm_calls"] == 0
    assert result["outcomes"]["rewrite_query"]["telemetry"]["llm_calls"] == 2
    assert result["outcomes"]["expand_context"]["telemetry"]["llm_calls"] == 1
    assert len(result["outcomes"]["expand_context"]["doc_ids"]) == 10
    if family == "retrieval_rank_dropout":
        assert len(result["initial"]["docs"]) == 3
    if family == "natural":
        assert result["initial"]["answer"] == "wrong"
        assert result["setup_telemetry"]["llm_calls"] == 0


def test_record_validation_rejects_duplicates_and_wrong_split():
    case = {"case_id": "a", "question_id": "a", "split": "train", "family": "vague_query"}
    row = {**case, "manifest_sha256": "hash", "outcomes": dict.fromkeys(ACTIONS)}
    study.validate_records([row], [case], "hash")
    with pytest.raises(ValueError):
        study.validate_records([row, row], [case], "hash")
    with pytest.raises(ValueError):
        study.validate_records([{**row, "split": "test"}], [case], "hash")


def test_study_lock_prevents_concurrent_collectors(tmp_path):
    with study.study_lock(tmp_path):
        with pytest.raises(RuntimeError, match="another process"):
            with study.study_lock(tmp_path):
                pass


@pytest.fixture
def offline_collection(tmp_path, monkeypatch, corpus):
    """Exercise the real collector/checkpoints while replacing all paid work."""
    source = [{"id": str(i), "question": f"Question {i}?", "gold": "France",
               "corrupted": {}, "baseline": {}} for i in range(5)]
    paths = [tmp_path / name for name in ("query.jsonl", "answer.jsonl", "corpus.jsonl")]
    for path, rows in zip(paths, (source, source, corpus)):
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setattr(study, "FIGURE_QUERY_SEED_PATHS", [paths[0]])
    monkeypatch.setattr(study, "FIGURE_ANSWER_PATH", paths[1])
    monkeypatch.setattr(study, "CORPUS_PATH", paths[2])
    monkeypatch.setattr(study, "OPENAI_API_KEY", "test-not-a-real-key")
    monkeypatch.setattr(study, "BudgetedLM", lambda *args, **kwargs: None)

    def mocked_case(case, source, corpus, retriever, lm):
        from test_reliability_analysis import case_row
        return case_row(case)

    monkeypatch.setattr(study, "run_case", mocked_case)
    directory = tmp_path / "study"
    directory.mkdir()
    return directory


def collect_offline(directory, **overrides):
    kwargs = dict(phase="development", workers=1, budget_usd=3, resume=False, limit=None)
    study.collect(directory, **(kwargs | overrides))


def test_collection_resume_freeze_and_heldout_phase(offline_collection):
    directory = offline_collection
    collect_offline(directory, limit=2)
    assert len(study.load_jsonl(directory / "development_outcomes.jsonl")) == 2
    with pytest.raises(ValueError, match="finish all"):
        study.fit(directory)
    with pytest.raises(FileExistsError, match="resume"):
        collect_offline(directory)
    collect_offline(directory, resume=True)
    assert len(study.load_jsonl(directory / "development_outcomes.jsonl")) == 8
    study.fit(directory)
    collect_offline(directory, phase="test")
    assert len(study.load_jsonl(directory / "test_outcomes.jsonl")) == 5
    assert (directory / "test_model_lock.json").exists()
    collect_offline(directory, phase="test", resume=True)
    assert len(study.load_jsonl(directory / "test_outcomes.jsonl")) == 5
    path = directory / "model.json"
    model = json.loads(path.read_text())
    model["policy"]["min_gain"] += .01
    path.write_text(json.dumps(model))
    with pytest.raises(ValueError, match="model changed"):
        collect_offline(directory, phase="test", resume=True)


def test_collection_preserves_completed_cases_after_failure(offline_collection, monkeypatch):
    original = study.run_case
    calls = []

    def fail_second(*args):
        calls.append(args[0]["case_id"])
        if len(calls) == 2:
            raise RuntimeError("simulated provider outage")
        return original(*args)

    monkeypatch.setattr(study, "run_case", fail_second)
    with pytest.raises(RuntimeError, match="preserved"):
        collect_offline(offline_collection)
    assert len(study.load_jsonl(offline_collection / "development_outcomes.jsonl")) == 1
    assert len(calls) == 2
    monkeypatch.setattr(study, "run_case", original)
    collect_offline(offline_collection, resume=True)
    assert len(study.load_jsonl(offline_collection / "development_outcomes.jsonl")) == 8


def test_collection_rejects_source_drift_before_calls(offline_collection):
    collect_offline(offline_collection, limit=1)
    path = study.FIGURE_ANSWER_PATH
    rows = study.load_jsonl(path)
    rows[0]["extra"] = "changed"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    with pytest.raises(ValueError, match="changed"):
        collect_offline(offline_collection, resume=True)


@pytest.mark.parametrize("command", ["collect", "fit", "report"])
def test_cli_dispatch_without_live_calls(tmp_path, monkeypatch, command):
    import reliability_analysis
    called = []
    monkeypatch.setattr(study, "collect", lambda *a, **k: called.append("collect"))
    monkeypatch.setattr(study, "fit", lambda *a, **k: called.append("fit"))
    monkeypatch.setattr(reliability_analysis, "create_report", lambda *a, **k: called.append("report"))
    monkeypatch.setattr(sys, "argv", ["study", command, "--directory", str(tmp_path)])
    study.main()
    assert called == [command]


def test_cli_rejects_excess_budget_before_work(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["study", "collect", "--budget-usd", "4"])
    with pytest.raises(SystemExit) as exc:
        study.main()
    assert exc.value.code == 2
