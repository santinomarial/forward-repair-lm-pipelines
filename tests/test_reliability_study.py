import copy
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
