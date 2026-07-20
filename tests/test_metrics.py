import pytest

from metrics import (
    all_support_recall_at_k,
    contains_answer,
    exact_match,
    recall_at_k,
    recovery_rate,
)


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("PARIS", "paris"),
        ("Paris!", "Paris"),
        ("Paris, France", "Paris France"),
        ("The Eiffel Tower", "Eiffel Tower"),
        ("  an   apple\n", "apple"),
        ("Yes, it was.", "yes"),
        ("No. That is incorrect.", "no"),
    ],
)
def test_exact_match_normalization(prediction, gold):
    assert exact_match(prediction, gold) == 1


@pytest.mark.parametrize(
    ("prediction", "gold"),
    [
        ("Paris, France", "Paris"),
        ("yesterday yes", "yes"),
        ("No", "yes"),
        ("a cat", "dog"),
    ],
)
def test_exact_match_rejects_non_matches(prediction, gold):
    assert exact_match(prediction, gold) == 0


def test_contains_answer_normalizes_case_punctuation_and_whitespace():
    assert contains_answer("It was NEW-YORK.\n", "new york") == 1
    assert contains_answer("It was Boston.", "new york") == 0


def test_recall_at_k_requires_any_support_document():
    docs = [{"id": "d1"}, {"id": "d2"}]

    assert recall_at_k(docs, ["d2", "d3"]) == 1
    assert recall_at_k(docs, ["d3", "d4"]) == 0


def test_all_support_at_k_requires_every_support_document():
    docs = [{"id": "d1"}, {"id": "d2"}, {"id": "d2"}]

    assert all_support_recall_at_k(docs, ["d1", "d2"]) == 1
    assert all_support_recall_at_k(docs, ["d1", "d3"]) == 0


def test_empty_support_set_is_vacuously_fully_recalled_but_not_partially_recalled():
    docs = [{"id": "d1"}]

    assert recall_at_k(docs, []) == 0
    assert all_support_recall_at_k(docs, []) == 1


def _metric_row(corrupted_em: int, repaired_em: int) -> dict:
    return {
        "corrupted": {"metrics": {"exact_match": corrupted_em}},
        "repaired": {"metrics": {"exact_match": repaired_em}},
    }


def test_recovery_rate_uses_only_corrupted_broken_examples():
    rows = [
        _metric_row(0, 1),
        _metric_row(0, 0),
        _metric_row(1, 1),
        _metric_row(1, 0),
    ]

    assert recovery_rate(rows) == {
        "corrupted_broken_count": 2,
        "repaired_fixed_count": 1,
        "recovery_rate": 0.5,
    }


def test_recovery_rate_handles_no_broken_examples():
    assert recovery_rate([_metric_row(1, 1)]) == {
        "corrupted_broken_count": 0,
        "repaired_fixed_count": 0,
        "recovery_rate": 0.0,
    }
