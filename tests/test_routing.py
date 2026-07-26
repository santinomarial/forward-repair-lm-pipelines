from dataclasses import replace
from types import SimpleNamespace

import pytest

from pipeline import ForwardRepairPipeline
from routing import (
    FailureSignals,
    HeuristicRepairPolicy,
    LexicalFailureDetector,
    RepairAction,
    RepairPolicy,
)


def _signals(**overrides) -> FailureSignals:
    base = FailureSignals(
        retrieval_score_margin=0.5,
        positive_score_fraction=1.0,
        question_document_overlap=0.8,
        query_document_overlap=0.8,
        answer_context_overlap=1.0,
        answer_in_context=True,
        answer_is_unknown=False,
        answer_is_binary=False,
    )
    return replace(base, **overrides)


def test_lexical_detector_extracts_backend_neutral_signals():
    detector = LexicalFailureDetector()
    signals = detector.detect(
        question="Who wrote Hamlet?",
        query="Hamlet author",
        docs=[
            {
                "id": "hamlet",
                "title": "Hamlet",
                "text": "William Shakespeare wrote Hamlet.",
                "score": 4.0,
            },
            {
                "id": "other",
                "title": "Other",
                "text": "An unrelated document.",
                "score": 1.0,
            },
        ],
        answer="William Shakespeare",
    )

    assert signals.retrieval_score_margin == pytest.approx(0.6)
    assert signals.positive_score_fraction == 1.0
    assert signals.question_document_overlap == 1.0
    assert signals.query_document_overlap == 0.5
    assert signals.answer_context_overlap == 1.0
    assert signals.answer_in_context is True
    assert signals.answer_is_unknown is False


def test_lexical_detector_handles_empty_retrieval_and_unknown_answer():
    signals = LexicalFailureDetector().detect(
        question="Who wrote Hamlet?",
        query="play author",
        docs=[],
        answer="UNKNOWN",
    )

    assert signals.retrieval_score_margin == 0.0
    assert signals.positive_score_fraction == 0.0
    assert signals.question_document_overlap == 0.0
    assert signals.answer_is_unknown is True
    assert signals.answer_in_context is False


@pytest.mark.parametrize(
    ("signals", "expected"),
    [
        (
            _signals(
                question_document_overlap=0.05,
                retrieval_score_margin=0.01,
            ),
            RepairAction.REPAIR_ITERATIVE,
        ),
        (
            _signals(
                question_document_overlap=0.15,
                retrieval_score_margin=0.01,
            ),
            RepairAction.REPAIR_QUERY,
        ),
        (
            _signals(
                answer_in_context=False,
                answer_context_overlap=0.0,
                answer_is_unknown=True,
            ),
            RepairAction.REPAIR_ANSWER,
        ),
        (
            _signals(
                answer_in_context=False,
                answer_context_overlap=0.2,
            ),
            RepairAction.REPAIR_ANSWER,
        ),
        (
            _signals(
                answer_in_context=False,
                answer_context_overlap=0.0,
                answer_is_binary=True,
            ),
            RepairAction.ACCEPT,
        ),
        (_signals(), RepairAction.ACCEPT),
    ],
)
def test_heuristic_policy_routes_expected_failure_class(signals, expected):
    assert HeuristicRepairPolicy().decide(signals) == expected


class FixedPolicy(RepairPolicy):
    name = "fixed_test"

    def __init__(self, action: RepairAction):
        self.action = action

    def decide(self, signals: FailureSignals) -> RepairAction:
        return self.action


class CountingRetriever:
    top_k = 1

    def __init__(self):
        self.calls = 0

    def retrieve(self, query: str, top_k: int | None = None) -> list[dict]:
        self.calls += 1
        return [
            {
                "id": query,
                "title": query.title(),
                "text": f"Evidence for {query}.",
                "score": 1.0,
            }
        ]

    def retrieve_union(self, queries: list[str], top_k_total: int) -> list[dict]:
        self.calls += len(queries)
        return self.retrieve(queries[0], top_k=top_k_total)


def _initial_run() -> dict:
    docs = [
        {
            "id": "initial",
            "title": "Initial",
            "text": "The retrieved evidence.",
            "score": 1.0,
        }
    ]
    return {
        "query": "initial",
        "docs": docs,
        "context": "Title: Initial\nText: The retrieved evidence.",
        "answer": "wrong",
        "telemetry": {
            "wall_clock_seconds": 0.6,
            "latency_seconds": {
                "query_generation": 0.1,
                "retrieval": 0.2,
                "answer_generation": 0.3,
            },
        },
    }


def test_adaptive_query_repair_reruns_retrieval_and_answer_generation():
    retriever = CountingRetriever()
    pipeline = ForwardRepairPipeline(retriever)
    pipeline.repaired_query_generator = lambda question, bad_query: SimpleNamespace(
        query="repaired"
    )
    pipeline.answer_generator = lambda question, context: SimpleNamespace(
        answer="correct"
    )

    result = pipeline.run_adaptive(
        "question",
        detector=LexicalFailureDetector(),
        policy=FixedPolicy(RepairAction.REPAIR_QUERY),
        initial_run=_initial_run(),
    )

    assert retriever.calls == 1
    assert result["query"] == "repaired"
    assert result["docs"][0]["id"] == "repaired"
    assert result["answer"] == "correct"
    assert result["routing"]["action"] == "repair_query"
    assert result["routing"]["repaired"] is True
    assert result["initial"]["answer"] == "wrong"
    assert result["telemetry"]["latency_seconds"]["retrieval"] >= 0.2


def test_adaptive_answer_repair_reuses_existing_retrieval():
    retriever = CountingRetriever()
    pipeline = ForwardRepairPipeline(retriever)
    pipeline.answer_repairer = (
        lambda question, context, bad_answer: SimpleNamespace(answer="grounded")
    )

    result = pipeline.run_adaptive(
        "question",
        detector=LexicalFailureDetector(),
        policy=FixedPolicy(RepairAction.REPAIR_ANSWER),
        initial_run=_initial_run(),
    )

    assert retriever.calls == 0
    assert result["query"] == "initial"
    assert result["docs"][0]["id"] == "initial"
    assert result["answer"] == "grounded"
    assert result["routing"]["action"] == "repair_answer"
    assert result["telemetry"]["latency_seconds"]["query_generation"] == 0.1
    assert result["telemetry"]["latency_seconds"]["retrieval"] == 0.2


def test_adaptive_accept_returns_initial_output_without_repair():
    retriever = CountingRetriever()
    pipeline = ForwardRepairPipeline(retriever)
    initial = _initial_run()

    result = pipeline.run_adaptive(
        "question",
        detector=LexicalFailureDetector(),
        policy=FixedPolicy(RepairAction.ACCEPT),
        initial_run=initial,
    )

    assert retriever.calls == 0
    assert result["answer"] == initial["answer"]
    assert result["routing"]["action"] == "accept"
    assert result["routing"]["repaired"] is False
