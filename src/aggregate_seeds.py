import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs"

SEED_FILES = [OUTPUT_DIR / f"hotpot_300_seed{s}_results.jsonl" for s in range(3)]
OUT_PATH = OUTPUT_DIR / "hotpot_300_aggregated.json"

MODES = ["baseline", "corrupted", "repaired"]
METRIC_KEYS = ["exact_match", "contains_answer", "recall_at_k", "all_support_recall_at_k"]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def seed_averages(rows: list[dict]) -> dict[str, dict[str, float]]:
    n = len(rows)
    totals: dict[str, dict[str, float]] = {m: {k: 0.0 for k in METRIC_KEYS} for m in MODES}
    for row in rows:
        for mode in MODES:
            for k in METRIC_KEYS:
                totals[mode][k] += row[mode]["metrics"][k]
    return {mode: {k: totals[mode][k] / n for k in METRIC_KEYS} for mode in MODES}


def mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mu = sum(values) / n
    variance = sum((v - mu) ** 2 for v in values) / (n - 1) if n > 1 else 0.0
    return mu, math.sqrt(variance)


def main() -> None:
    per_seed: list[dict[str, dict[str, float]]] = []
    for path in SEED_FILES:
        rows = load_jsonl(path)
        per_seed.append(seed_averages(rows))
        print(f"  Loaded {len(rows)} rows from {path.name}")

    aggregated: dict[str, dict[str, dict]] = {}
    for mode in MODES:
        aggregated[mode] = {}
        for k in METRIC_KEYS:
            vals = [per_seed[s][mode][k] for s in range(len(SEED_FILES))]
            mu, sd = mean_std(vals)
            aggregated[mode][k] = {"mean": mu, "std": sd, "per_seed": vals}

    # markdown table
    col_labels = ["EM", "Contains", "Recall@K", "AllSupport@K"]
    header = f"| {'Condition':<10} | " + " | ".join(f"{'  ' + c + '  ':^20}" for c in col_labels) + " |"
    sep = "|" + "|".join("-" * (len(s) + 2) for s in header.split("|")[1:-1]) + "|"

    print(f"\n### Aggregated Results (mean ± std across {len(SEED_FILES)} seeds, n=300 each)\n")
    print(header)
    print(sep)
    for mode in MODES:
        cells = []
        for k in METRIC_KEYS:
            mu = aggregated[mode][k]["mean"]
            sd = aggregated[mode][k]["std"]
            cells.append(f"{mu:.2f} ± {sd:.2f}")
        print(f"| {mode:<10} | " + " | ".join(f"{c:^20}" for c in cells) + " |")

    OUT_PATH.write_text(json.dumps(aggregated, indent=2))
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
