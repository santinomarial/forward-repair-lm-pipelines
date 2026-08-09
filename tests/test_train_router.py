from dataclasses import replace

import pytest

from routing import FailureSignals, RepairAction
from train_router import (
    RouterExample,
    build_training_examples,
    classification_metrics,
    fit_policy,
    repeated_holdout_metrics,
    split_by_question,
)


def _run(answer: str, exact_match: int = 0) -> dict:
    return {
        "query": "alpha",
        "docs": [
            {
                "id": "alpha",
                "title": "Alpha",
                "text": "Alpha evidence supports the answer.",
                "score": 1.0,
            }
        ],
        "answer": answer,
        "metrics": {"exact_match": exact_match},
    }


def test_training_examples_use_controlled_stage_labels():
    query_rows = [
        {
            "id": "q1",
            "question": "What is alpha?",
            "baseline": _run("alpha", exact_match=1),
            "corrupted": _run("UNKNOWN"),
            "repaired": _run("wrong"),
            "repaired_iterative": _run("alpha", exact_match=1),
        },
        {
            "id": "q2",
            "question": "What is beta?",
            "baseline": _run("beta", exact_match=1),
            "corrupted": _run("UNKNOWN"),
            "repaired": _run("beta", exact_match=1),
            "repaired_iterative": _run("beta", exact_match=1),
        },
    ]
    answer_rows = [
        {
            "id": "q1",
            "question": "What is alpha?",
            "corrupted": _run("hallucination"),
        }
    ]

    examples = build_training_examples(query_rows, answer_rows)

    assert [(example.source, example.target) for example in examples] == [
        ("baseline", RepairAction.ACCEPT),
        ("query_corrupted", RepairAction.REPAIR_QUERY),
        ("baseline", RepairAction.ACCEPT),
        ("query_corrupted", RepairAction.REPAIR_QUERY),
        ("answer_corrupted", RepairAction.REPAIR_ANSWER),
    ]


def test_split_keeps_all_question_variants_together():
    examples = [
        RouterExample(
            question_id=f"q{index // 3}",
            source=str(index % 3),
            signals=_signal_for(RepairAction.ACCEPT),
            target=RepairAction.ACCEPT,
        )
        for index in range(30)
    ]

    train, test, _ = split_by_question(examples, test_fraction=0.2, seed=7)

    train_ids = {example.question_id for example in train}
    test_ids = {example.question_id for example in test}
    assert train_ids.isdisjoint(test_ids)
    assert len(test_ids) == 2


def _signal_for(action: RepairAction) -> FailureSignals:
    base = FailureSignals(
        retrieval_score_margin=0.8,
        positive_score_fraction=1.0,
        question_document_overlap=0.8,
        query_document_overlap=0.8,
        answer_context_overlap=1.0,
        answer_in_context=True,
        answer_is_unknown=False,
        answer_is_binary=False,
    )
    if action == RepairAction.REPAIR_QUERY:
        return replace(base, question_document_overlap=0.2)
    if action == RepairAction.REPAIR_ANSWER:
        return replace(
            base,
            answer_context_overlap=0.0,
            answer_in_context=False,
            answer_is_unknown=True,
        )
    if action == RepairAction.REPAIR_ITERATIVE:
        return replace(
            base,
            retrieval_score_margin=0.0,
            positive_score_fraction=0.0,
            question_document_overlap=0.0,
            query_document_overlap=0.0,
        )
    return base


def test_softmax_training_learns_separable_actions():
    examples = [
        RouterExample(
            question_id=f"{action.value}-{index}",
            source="synthetic",
            signals=_signal_for(action),
            target=action,
        )
        for action in (
            RepairAction.ACCEPT,
            RepairAction.REPAIR_QUERY,
            RepairAction.REPAIR_ANSWER,
        )
        for index in range(20)
    ]

    policy = fit_policy(
        examples,
        epochs=1_000,
        learning_rate=0.1,
        l2=0.001,
    )
    metrics = classification_metrics(policy, examples)

    assert metrics["accuracy"] >= 0.95
    assert metrics["macro_f1"] >= 0.95
    assert metrics["predicted_repair_rate"] == pytest.approx(2 / 3)


def test_repeated_holdout_reports_each_seed():
    examples = [
        RouterExample(
            question_id=f"{action.value}-{index}",
            source="synthetic",
            signals=_signal_for(action),
            target=action,
        )
        for action in (
            RepairAction.ACCEPT,
            RepairAction.REPAIR_QUERY,
            RepairAction.REPAIR_ANSWER,
        )
        for index in range(20)
    ]

    metrics = repeated_holdout_metrics(
        examples,
        seeds=[1, 2],
        test_fraction=0.2,
        epochs=500,
        learning_rate=0.1,
        l2=0.001,
        class_weight_power=0.5,
    )

    assert metrics["seeds"] == [1, 2]
    assert len(metrics["runs"]) == 2
    assert metrics["accuracy"]["mean"] >= 0.9
