"""Paired bootstrap comparisons for forward-repair experiment outcomes."""

import argparse
import json
from pathlib import Path

import numpy as np

from config import FIGURE_ITERATIVE_PATH, SIGNIFICANCE_PATH


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _interval(samples: np.ndarray, confidence: float) -> list[float]:
    tail = (1.0 - confidence) / 2.0
    return [
        float(np.quantile(samples, tail)),
        float(np.quantile(samples, 1.0 - tail)),
    ]


def _paired_bootstrap_means(
    arrays: list[np.ndarray],
    *,
    n_resamples: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    n = len(arrays[0])
    if n == 0:
        raise ValueError("bootstrap comparison requires at least one eligible row")
    if any(len(array) != n for array in arrays):
        raise ValueError("paired arrays must have equal length")

    samples = [np.empty(n_resamples, dtype=float) for _ in arrays]
    for start in range(0, n_resamples, 1_000):
        size = min(1_000, n_resamples - start)
        indices = rng.integers(0, n, size=(size, n))
        for output, values in zip(samples, arrays):
            output[start : start + size] = values[indices].mean(axis=1)
    return samples


def bootstrap_comparisons(
    rows: list[dict],
    *,
    n_resamples: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive")
    if not rows:
        raise ValueError("input contains no rows")

    rng = np.random.default_rng(seed)
    corrupted = np.array(
        [row["corrupted"]["metrics"]["exact_match"] for row in rows],
        dtype=float,
    )
    repaired = np.array(
        [row["repaired"]["metrics"]["exact_match"] for row in rows],
        dtype=float,
    )
    repaired_iterative = np.array(
        [row["repaired_iterative"]["metrics"]["exact_match"] for row in rows],
        dtype=float,
    )

    em_samples = _paired_bootstrap_means(
        [corrupted, repaired, repaired - corrupted],
        n_resamples=n_resamples,
        rng=rng,
    )

    broken = corrupted == 0
    single_recovered = repaired[broken]
    iterative_recovered = repaired_iterative[broken]
    recovery_samples = _paired_bootstrap_means(
        [
            single_recovered,
            iterative_recovered,
            iterative_recovered - single_recovered,
        ],
        n_resamples=n_resamples,
        rng=rng,
    )

    def estimate(values: np.ndarray, samples: np.ndarray) -> dict:
        return {
            "estimate": float(values.mean()),
            "ci": _interval(samples, confidence),
        }

    return {
        "method": "paired percentile bootstrap",
        "confidence": confidence,
        "n_resamples": n_resamples,
        "seed": seed,
        "corrupted_vs_repaired_em": {
            "n": len(rows),
            "corrupted": estimate(corrupted, em_samples[0]),
            "repaired": estimate(repaired, em_samples[1]),
            "difference_repaired_minus_corrupted": estimate(
                repaired - corrupted,
                em_samples[2],
            ),
        },
        "single_shot_vs_iterative_recovery": {
            "n_corrupted_broken": int(broken.sum()),
            "single_shot": estimate(single_recovered, recovery_samples[0]),
            "iterative": estimate(iterative_recovered, recovery_samples[1]),
            "difference_iterative_minus_single": estimate(
                iterative_recovered - single_recovered,
                recovery_samples[2],
            ),
        },
    }


def _percent(value: float) -> str:
    return f"{value:.1%}"


def print_results(results: dict) -> None:
    print("Paired bootstrap significance analysis")
    print(
        f"  {results['n_resamples']:,} resamples, "
        f"{results['confidence']:.0%} confidence intervals"
    )
    for label, comparison in [
        ("Corrupted vs repaired EM", results["corrupted_vs_repaired_em"]),
        (
            "Single-shot vs iterative recovery",
            results["single_shot_vs_iterative_recovery"],
        ),
    ]:
        difference_key = next(key for key in comparison if key.startswith("difference_"))
        difference = comparison[difference_key]
        print(
            f"  {label}: Δ={_percent(difference['estimate'])} "
            f"CI [{_percent(difference['ci'][0])}, {_percent(difference['ci'][1])}]"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=FIGURE_ITERATIVE_PATH)
    parser.add_argument("--output", type=Path, default=SIGNIFICANCE_PATH)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    results = bootstrap_comparisons(
        load_jsonl(args.input),
        n_resamples=args.resamples,
        confidence=args.confidence,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print_results(results)
    print(f"  Saved to {args.output}")


if __name__ == "__main__":
    main()
