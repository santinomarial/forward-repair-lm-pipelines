"""Re-score stored results with the updated exact_match — no LM calls."""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from metrics import exact_match

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = ROOT / "outputs" / "hotpot_50_final_results.jsonl"

_ALL_MODES = ["baseline", "corrupted", "repaired", "repaired_iterative"]


def rescore_row(row: dict) -> dict:
    gold = row["gold"]
    for mode in [m for m in _ALL_MODES if m in row]:
        pred = row[mode]["answer"]
        row[mode]["metrics"]["exact_match"] = exact_match(pred, gold)
    return row


def summarize(rows: list[dict]) -> dict:
    metric_keys = ["exact_match", "contains_answer", "recall_at_k", "all_support_recall_at_k"]
    modes = [m for m in _ALL_MODES if m in rows[0]]
    totals = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for mode in modes:
            for k in metric_keys:
                totals[mode][k] += row[mode]["metrics"][k]

    n = len(rows)
    summary = {
        mode: {k: totals[mode][k] / n for k in metric_keys}
        for mode in modes
    }

    corrupted_broken = repaired_fixed = 0
    for row in rows:
        if row["corrupted"]["metrics"]["exact_match"] == 0:
            corrupted_broken += 1
            if row["repaired"]["metrics"]["exact_match"] == 1:
                repaired_fixed += 1

    summary["recovery"] = {
        "corrupted_broken_count": corrupted_broken,
        "repaired_fixed_count": repaired_fixed,
        "recovery_rate": repaired_fixed / corrupted_broken if corrupted_broken else 0.0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    in_path: Path = args.input
    out_results: Path = args.output or in_path.with_name(in_path.stem + "_renormalized.jsonl")
    out_summary: Path = out_results.with_name(out_results.stem.replace("_results", "") + "_summary_renormalized.json")

    rows = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]
    rows = [rescore_row(row) for row in rows]

    out_results.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    summary = summarize(rows)
    out_summary.write_text(json.dumps(summary, indent=2) + "\n")

    modes = [m for m in _ALL_MODES if m in summary]
    print(f"Rescored {len(rows)} rows.")
    print(f"Results → {out_results}")
    for mode in modes:
        m = summary[mode]
        print(
            f"  {mode:10s}  EM={m['exact_match']:.3f}  "
            f"Contains={m['contains_answer']:.3f}  "
            f"Recall@K={m['recall_at_k']:.3f}  "
            f"AllSupport@K={m['all_support_recall_at_k']:.3f}"
        )
    rec = summary["recovery"]
    print(f"  recovery: {rec['repaired_fixed_count']}/{rec['corrupted_broken_count']} "
          f"= {rec['recovery_rate']:.2%}")


if __name__ == "__main__":
    main()
