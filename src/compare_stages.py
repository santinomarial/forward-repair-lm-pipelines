"""Compare query-corruption vs answer-corruption results in a single table."""
import argparse
import json
import math
from pathlib import Path

from config import STAGE_COMPARISON_PATH


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))


def seed_summary(rows: list[dict]) -> dict:
    n = len(rows)
    em = {
        mode: sum(r[mode]["metrics"]["exact_match"] for r in rows) / n
        for mode in ("baseline", "corrupted", "repaired")
    }
    broken = [r for r in rows if r["corrupted"]["metrics"]["exact_match"] == 0]
    fixed  = [r for r in broken if r["repaired"]["metrics"]["exact_match"] == 1]
    rate   = len(fixed) / len(broken) if broken else float("nan")
    return {
        "baseline_em":  em["baseline"],
        "corrupted_em": em["corrupted"],
        "repaired_em":  em["repaired"],
        "recovery_rate": rate,
        "n": n,
    }


def fmt(mu: float, sd: float) -> str:
    if mu != mu:
        return "nan"
    if sd == 0.0:
        return f"{mu:.3f}      "
    return f"{mu:.3f} ± {sd:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-inputs", required=True,
                        help="Comma-separated result files for query-corruption runs")
    parser.add_argument("--answer-inputs", required=True,
                        help="Comma-separated result files for answer-corruption runs")
    parser.add_argument("--output", type=Path,
                        default=STAGE_COMPARISON_PATH)
    args = parser.parse_args()

    def load_stage(csv: str) -> list[dict]:
        paths = [Path(p.strip()) for p in csv.split(",")]
        per_seed = []
        for p in paths:
            rows = load_jsonl(p)
            per_seed.append(seed_summary(rows))
            print(f"  Loaded {rows[0].get('id','?')[:8]}… ({len(rows)} rows) from {p.name}")
        return per_seed

    print("Query-corruption files:")
    q_seeds = load_stage(args.query_inputs)
    print("Answer-corruption files:")
    a_seeds = load_stage(args.answer_inputs)

    cols = ["baseline_em", "corrupted_em", "repaired_em", "recovery_rate"]
    col_labels = ["Baseline EM", "Corrupted EM", "Repaired EM", "Recovery Rate"]

    CW = 16
    header = f"{'Stage':<22}  " + "  ".join(f"{c:^{CW}}" for c in col_labels)
    sep = "-" * len(header)

    print(f"\n### Stage Comparison  (mean ± std across seeds)\n")
    print(header)
    print(sep)

    output = {}
    for label, seeds in [("query-corruption", q_seeds), ("answer-corruption", a_seeds)]:
        row_cells = []
        output[label] = {}
        for col in cols:
            vals = [s[col] for s in seeds]
            mu, sd = _mean(vals), _std(vals)
            output[label][col] = {"mean": mu, "std": sd, "per_seed": vals}
            row_cells.append(f"{fmt(mu, sd):^{CW}}")
        n_mean = _mean([s["n"] for s in seeds])
        print(f"{label:<22}  " + "  ".join(row_cells) + f"  (n≈{n_mean:.0f}/seed, {len(seeds)} seed{'s' if len(seeds)>1 else ''})")

    args.output.write_text(json.dumps(output, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
