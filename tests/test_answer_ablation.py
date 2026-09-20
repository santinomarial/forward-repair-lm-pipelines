import copy
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

import answer_ablation
from answer_ablation import (
    CONDITIONS,
    build_modules,
    prepare_output,
    run_example,
    summarize_ablation,
    validate_sources,
)
from dspy_modules import AnswerGenerator, AnswerRepairer, BlindAnswerRepairer, RepairAnswer


@pytest.fixture
def source():
    return {
        "id": "q1",
        "question": "Where was Ada born?",
        "gold": "London",
        "corrupted": {
            "query": "Ada birthplace",
            "docs": [{"id": "ada", "title": "Ada", "text": "Ada was born in London."}],
            "context": "Title: Ada\nText: Ada was born in London.",
            "answer": "Paris",
            "metrics": {"exact_match": 1},  # Deliberately stale; never trusted.
        },
    }


def _fake_modules(calls, model):
    def make(mode):
        def predict(**kwargs):
            calls.append((mode, kwargs))
            model.history.append({
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "cost": 0.001,
            })
            return SimpleNamespace(answer="London")
        return predict
    return {mode: make(mode) for mode in CONDITIONS}


def test_run_reuses_evidence_and_exposes_bad_answer_only_to_revision(source):
    original = copy.deepcopy(source)
    calls = []
    model = SimpleNamespace(history=[])
    row = run_example(source, _fake_modules(calls, model), [model], seed=7)

    assert source == original
    assert len(calls) == 3
    assert {mode for mode, _ in calls} == set(CONDITIONS)
    for mode, kwargs in calls:
        assert kwargs["context"] == source["corrupted"]["context"]
        assert kwargs["question"] == source["question"]
        assert set(kwargs) == {"question", "context"} | ({"bad_answer"} if mode == "revision" else set())
        if mode == "revision":
            assert kwargs["bad_answer"] == "Paris"
        telemetry = row[mode]["telemetry"]
        assert telemetry["llm_calls"] == 1
        assert telemetry["total_tokens"] == 12
        assert telemetry["latency_seconds"]["query_generation"] == 0
        assert telemetry["latency_seconds"]["retrieval"] == 0
    assert row["corrupted"]["metrics"]["exact_match"] == 0
    assert row["evidence"]["docs"] == source["corrupted"]["docs"]
    assert row["evidence"]["query"] == source["corrupted"]["query"]
    repeated = run_example(source, _fake_modules([], model), [model], seed=7)
    assert row["condition_order"] == repeated["condition_order"]


def test_matched_revision_preserves_instructions_and_removes_only_bad_answer():
    modules = build_modules()
    original = modules["revision"].generate.signature
    blind = modules["blind_revision"].generate.signature
    assert original.instructions == blind.instructions == RepairAnswer.instructions
    assert set(original.input_fields) == {"question", "context", "bad_answer"}
    assert set(blind.input_fields) == {"question", "context"}
    for field in ("question", "context", "answer"):
        assert original.fields[field].json_schema_extra == blind.fields[field].json_schema_extra


@pytest.mark.parametrize("module_class", [AnswerGenerator, AnswerRepairer, BlindAnswerRepairer])
def test_all_answer_modules_keep_existing_evidence_guard(module_class):
    module = module_class()
    captured = {}

    def predict(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(answer="London")

    module.generate = predict
    kwargs = {"question": "Where?", "context": "Evidence"}
    if module_class is AnswerRepairer:
        kwargs["bad_answer"] = "Paris"
    module(**kwargs)
    assert captured["context"] == (
        "Rules:\n"
        "1. Use only the retrieved context below.\n"
        "2. Do not use outside knowledge.\n"
        "3. If the context does not explicitly support the answer, output UNKNOWN.\n\n"
        "Evidence"
    )


def _scored_rows():
    rows = []
    for index, values in enumerate([(0, 1, 1, 1), (0, 0, 1, 0), (1, 0, 1, 1), (1, 1, 1, 0)]):
        row = {"id": str(index), "gold": "London"}
        for mode, correct in zip(("corrupted", *CONDITIONS), values):
            row[mode] = {
                "answer": "London" if correct else "UNKNOWN",
                "metrics": {"exact_match": 99},
                "telemetry": {
                    "llm_calls": 1, "prompt_tokens": 10, "completion_tokens": 2,
                    "total_tokens": 12, "estimated_cost_usd": 0.001, "priced_calls": 1,
                    "wall_clock_seconds": 1.0,
                    "latency_seconds": {"query_generation": 0, "retrieval": 0, "answer_generation": 1},
                },
            }
        rows.append(row)
    return rows


def test_analysis_measures_recovery_damage_and_paired_differences():
    rows = _scored_rows()
    result = summarize_ablation(rows, n_resamples=200, seed=2)
    assert result == summarize_ablation(rows, n_resamples=200, seed=2)
    revision = result["conditions"]["revision"]
    assert revision["exact_match"] == 0.5
    assert revision["recovery_rate"] == 0.5
    assert revision["damage_rate"] == 0.5
    assert revision["recovered_count"] == revision["damaged_count"] == 1
    assert revision["abstention_rate"] == 0.5
    assert result["conditions"]["blind_revision"]["damage_rate"] == 0
    primary = result["comparisons"]["blind_revision_minus_revision"]
    assert primary["exact_match"]["estimate"] == 0.5
    assert primary["recovery"]["estimate"] == 0.5
    assert primary["recovery"]["n"] == 2
    usage = result["instrumentation"]["fresh"]
    assert usage["tokens"]["total"] == 48
    assert usage["llm_calls"]["total"] == 4
    assert usage["priced_calls"] == 4
    assert usage["wall_clock_seconds"]["p95"] == 1


@pytest.mark.parametrize("corrupted_answer", ["London", "wrong"])
def test_empty_eligible_stratum_is_undefined_not_zero(corrupted_answer):
    rows = _scored_rows()
    for row in rows:
        row["corrupted"]["answer"] = corrupted_answer
    result = summarize_ablation(rows, n_resamples=10)
    if corrupted_answer == "London":
        assert result["conditions"]["fresh"]["recovery_rate"] is None
        assert result["comparisons"]["fresh_minus_revision"]["recovery"] == {
            "n": 0, "estimate": None, "ci": None,
        }
    else:
        assert result["conditions"]["fresh"]["damage_rate"] is None
    json.dumps(result, allow_nan=False)


def test_posthoc_format_audit_distinguishes_exact_match_from_answer_containment():
    rows = _scored_rows()
    for row in rows:
        row["revision"]["answer"] = "Ada was born in London."
    report = summarize_ablation(rows, n_resamples=10)
    audit = report["response_format_audit"]
    assert audit["blind_only_exact_matches"] == 4
    assert audit["revision_only_exact_matches"] == 0
    assert audit["blind_gains_where_revision_contains_gold"] == 4
    assert "post-hoc" in audit["status"]


def test_reject_empty_duplicate_or_mixed_run_analysis():
    with pytest.raises(ValueError, match="no rows"):
        summarize_ablation([])
    rows = _scored_rows()
    with pytest.raises(ValueError, match="duplicate"):
        summarize_ablation([rows[0], rows[0]])
    rows[0]["run_config"] = {"model": "one"}
    rows[1]["run_config"] = {"model": "two"}
    with pytest.raises(ValueError, match="different run"):
        summarize_ablation(rows, n_resamples=10)


def test_source_validation(source):
    validate_sources([source])
    with pytest.raises(ValueError, match="no rows"):
        validate_sources([])
    with pytest.raises(ValueError, match="duplicate"):
        validate_sources([source, source])
    source["corrupted"]["context"] = None
    with pytest.raises(ValueError, match="context must be text"):
        validate_sources([source])


def test_output_safety_and_resumption(tmp_path, source):
    path = tmp_path / "run.jsonl"
    config = {"model": "fake", "input_sha256": "abc"}
    assert prepare_output(path, [source], config, resume=False) == []
    with pytest.raises(FileExistsError, match="already exists"):
        prepare_output(path, [source], config, resume=False)
    row = {"id": source["id"], "run_config": config, **dict.fromkeys(CONDITIONS, {})}
    path.write_text(json.dumps(row) + "\n")
    assert prepare_output(path, [source], config, resume=True) == [row]
    with pytest.raises(ValueError, match="same input"):
        prepare_output(path, [source], {"model": "different"}, resume=True)
    with pytest.raises(ValueError, match="fewer examples"):
        prepare_output(path, [], config, resume=True)
    del row["fresh"]
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="incomplete condition"):
        prepare_output(path, [source], config, resume=True)
    path.write_text('{"partial":')
    with pytest.raises(ValueError, match="incomplete JSONL"):
        prepare_output(path, [source], config, resume=True)


def test_analyze_only_never_initializes_llm(tmp_path, monkeypatch):
    source = tmp_path / "run.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in _scored_rows()))
    summary = tmp_path / "summary.json"
    monkeypatch.setattr(answer_ablation, "experiment_paths", lambda _: (tmp_path / "unused", summary))

    def forbidden(*args, **kwargs):
        pytest.fail("offline analysis must not initialize an LLM")

    monkeypatch.setattr(answer_ablation, "build_llm_backend", forbidden)
    monkeypatch.setattr(sys, "argv", ["answer_ablation", "--analyze-only", str(source), "--resamples", "10"])
    answer_ablation.main()
    assert json.loads(summary.read_text())["examples"] == 4


def test_live_cli_and_resume_with_mock_lm(tmp_path, monkeypatch, source):
    input_path = tmp_path / "source.jsonl"
    input_path.write_text(json.dumps(source) + "\n")
    results = tmp_path / "results.jsonl"
    summary = tmp_path / "summary.json"
    calls = []
    models_created = []
    model = SimpleNamespace(history=[])

    def create_lm(**kwargs):
        models_created.append(kwargs)
        return model

    def backend(*args, **kwargs):
        assert kwargs["cache"] is False
        return SimpleNamespace(create_lm=create_lm)

    original_modules = build_modules

    def modules():
        result = original_modules()
        for mode, module in result.items():
            mock = _fake_modules(calls, model)[mode]
            mock.signature = module.generate.signature
            module.generate = mock
        return result

    monkeypatch.setattr(answer_ablation, "build_modules", modules)
    monkeypatch.setattr(answer_ablation, "build_llm_backend", backend)
    monkeypatch.setattr(answer_ablation.dspy, "context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(answer_ablation, "experiment_paths", lambda _: (results, summary))
    argv = ["answer_ablation", "--input", str(input_path), "--max-examples", "1", "--resamples", "10"]
    monkeypatch.setattr(sys, "argv", argv)
    answer_ablation.main()
    saved = results.read_text()
    report = json.loads(summary.read_text())
    assert report["complete"] is True
    assert report["run_config"]["lm_cache"] is False
    assert report["run_config"]["temperature"] == 0
    assert set(report["run_config"]["prompt_sha256"]) == set(CONDITIONS)
    assert models_created == [{"temperature": 0, "seed": 0}]
    assert len(calls) == 3

    monkeypatch.setattr(sys, "argv", [*argv, "--resume"])
    answer_ablation.main()
    assert results.read_text() == saved
    assert len(calls) == 3
    assert len(models_created) == 1
    input_path.write_text(json.dumps({**source, "gold": "different"}) + "\n")
    with pytest.raises(ValueError, match="same input"):
        answer_ablation.main()
    assert len(calls) == 3


def test_partial_run_summary_is_explicit():
    rows = _scored_rows()
    for row in rows:
        row["run_config"] = {"examples": 300}
    assert summarize_ablation(rows, n_resamples=10)["complete"] is False


@pytest.mark.parametrize("arguments", [
    ["--max-examples", "0"],
    ["--max-tokens", "0"],
    ["--resamples", "0"],
    ["--confidence", "1"],
    ["--output-suffix", "../source"],
])
def test_cli_rejects_unsafe_arguments(arguments, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["answer_ablation", *arguments])
    with pytest.raises(SystemExit) as exc:
        answer_ablation.main()
    assert exc.value.code == 2


@pytest.mark.parametrize("field,value", [("question", ""), ("docs", "not a list")])
def test_source_validation_rejects_bad_fields(source, field, value):
    if field == "question":
        source[field] = value
    else:
        source["corrupted"][field] = value
    with pytest.raises(ValueError):
        validate_sources([source])


@pytest.mark.parametrize("analyze_only", [True, False])
def test_cli_rejects_overwriting_source(tmp_path, monkeypatch, analyze_only):
    path = tmp_path / "source.jsonl"
    monkeypatch.setattr(answer_ablation, "experiment_paths", lambda _: (tmp_path / "results", path))
    flag = "--analyze-only" if analyze_only else "--input"
    monkeypatch.setattr(sys, "argv", ["answer_ablation", flag, str(path)])
    with pytest.raises(SystemExit) as exc:
        answer_ablation.main()
    assert exc.value.code == 2
