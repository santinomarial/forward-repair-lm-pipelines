import pytest

from significance import bootstrap_comparisons


def _row(corrupted: int, repaired: int, iterative: int) -> dict:
    return {
        "corrupted": {"metrics": {"exact_match": corrupted}},
        "repaired": {"metrics": {"exact_match": repaired}},
        "repaired_iterative": {"metrics": {"exact_match": iterative}},
    }


def test_bootstrap_comparisons_use_paired_rows_and_broken_subset():
    rows = [
        _row(0, 0, 1),
        _row(0, 1, 1),
        _row(0, 0, 0),
        _row(1, 1, 0),
    ]

    result = bootstrap_comparisons(rows, n_resamples=500, seed=7)

    em = result["corrupted_vs_repaired_em"]
    recovery = result["single_shot_vs_iterative_recovery"]
    assert em["corrupted"]["estimate"] == 0.25
    assert em["repaired"]["estimate"] == 0.5
    assert em["difference_repaired_minus_corrupted"]["estimate"] == 0.25
    assert recovery["n_corrupted_broken"] == 3
    assert recovery["single_shot"]["estimate"] == pytest.approx(1 / 3)
    assert recovery["iterative"]["estimate"] == pytest.approx(2 / 3)
    assert recovery["difference_iterative_minus_single"]["estimate"] == pytest.approx(
        1 / 3
    )


def test_bootstrap_is_reproducible_and_validates_input():
    rows = [_row(0, 1, 1), _row(0, 0, 1)]
    first = bootstrap_comparisons(rows, n_resamples=50, seed=11)
    second = bootstrap_comparisons(rows, n_resamples=50, seed=11)
    assert first == second

    with pytest.raises(ValueError, match="no rows"):
        bootstrap_comparisons([])
    with pytest.raises(ValueError, match="at least one eligible"):
        bootstrap_comparisons([_row(1, 1, 1)])
