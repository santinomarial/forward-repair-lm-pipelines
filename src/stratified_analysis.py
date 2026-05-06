import argparse
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = ROOT / "outputs" / "hotpot_50_final_results_renormalized.jsonl"
CORPUS_PATH = ROOT / "data" / "hotpot_corpus.jsonl"
OUTPUT_PATH = ROOT / "outputs" / "stratified_analysis.json"
MULTI_OUTPUT_PATH = ROOT / "outputs" / "stratified_300x3.json"

_ALL_MODES = ["baseline", "corrupted", "repaired", "repaired_iterative"]
MODES = _ALL_MODES[:3]  # default; overridden from data in main()
METRIC_KEYS = ["exact_match", "contains_answer", "recall_at_k", "all_support_recall_at_k"]
STRATA = ["overall", "single_hop", "multi_hop", "yesno", "bridge"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def classify(gold: str, support_doc_ids: list[str], corpus: dict[str, str]) -> str:
    if gold.lower().strip() in ("yes", "no"):
        return "genuinely-multi-hop"
    norm_gold = normalize(gold)
    matches = sum(
        1 for doc_id in support_doc_ids
        if norm_gold in normalize(corpus.get(doc_id, ""))
    )
    if matches == 0:
        return "answer-not-in-support"
    if matches == 1:
        return "single-hop-sufficient"
    return "genuinely-multi-hop"


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _std(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mu = _mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (n - 1))


def compute_metrics(rows: list[dict], mode: str) -> dict:
    base = {k: _mean([r[mode]["metrics"][k] for r in rows]) for k in METRIC_KEYS}
    em_full = [
        r[mode]["metrics"]["exact_match"]
        for r in rows
        if r[mode]["metrics"]["all_support_recall_at_k"] == 1
    ]
    base["em_given_full_retrieval"] = _mean(em_full)
    base["em_given_full_retrieval_n"] = len(em_full)
    return base


def recovery_rate(rows: list[dict], repair_mode: str = "repaired") -> dict:
    broken = [r for r in rows if r["corrupted"]["metrics"]["exact_match"] == 0]
    fixed = [r for r in broken if r[repair_mode]["metrics"]["exact_match"] == 1]
    return {
        "corrupted_broken": len(broken),
        "repaired_fixed": len(fixed),
        "rate": len(fixed) / len(broken) if broken else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-seed stats
# ---------------------------------------------------------------------------

def compute_seed_stats(rows: list[dict], corpus: dict[str, str], modes: list[str]) -> dict:
    hop_classes = {
        r["id"]: classify(r["gold"], r["support_doc_ids"], corpus) for r in rows
    }
    excluded   = [r for r in rows if hop_classes[r["id"]] == "answer-not-in-support"]
    single_hop = [r for r in rows if hop_classes[r["id"]] == "single-hop-sufficient"]
    multi_hop  = [r for r in rows if hop_classes[r["id"]] == "genuinely-multi-hop"]
    included   = single_hop + multi_hop
    yesno      = [r for r in multi_hop if r["gold"].lower().strip() in ("yes", "no")]
    bridge     = [r for r in multi_hop if r["gold"].lower().strip() not in ("yes", "no")]

    strata_rows = {
        "overall":    included,
        "single_hop": single_hop,
        "multi_hop":  multi_hop,
        "yesno":      yesno,
        "bridge":     bridge,
    }

    # Recovery rates for each repair mode present
    repair_modes = [m for m in modes if m not in ("baseline", "corrupted")]
    recovery = {}
    for stratum_key, stratum_list in [
        ("overall", included), ("single_hop", single_hop), ("multi_hop", multi_hop)
    ]:
        recovery[stratum_key] = {
            rm: recovery_rate(stratum_list, repair_mode=rm) for rm in repair_modes
        }

    return {
        "modes": modes,
        "n": {
            "total":      len(rows),
            "excluded":   len(excluded),
            "single_hop": len(single_hop),
            "multi_hop":  len(multi_hop),
            "yesno":      len(yesno),
            "bridge":     len(bridge),
        },
        "metrics": {
            stratum: {mode: compute_metrics(strata_rows[stratum], mode) for mode in modes}
            for stratum in STRATA
        },
        "recovery": recovery,
    }


# ---------------------------------------------------------------------------
# Aggregation (multi-seed)
# ---------------------------------------------------------------------------

def aggregate_seeds(per_seed: list[dict]) -> dict:
    modes = per_seed[0]["modes"]
    repair_modes = [m for m in modes if m not in ("baseline", "corrupted")]

    agg_metrics: dict = {}
    for stratum in STRATA:
        agg_metrics[stratum] = {}
        for mode in modes:
            agg_metrics[stratum][mode] = {}
            for k in METRIC_KEYS + ["em_given_full_retrieval"]:
                vals = [s["metrics"][stratum][mode].get(k, float("nan")) for s in per_seed]
                valid = [v for v in vals if v == v]
                agg_metrics[stratum][mode][k] = {
                    "mean": _mean(valid),
                    "std":  _std(valid),
                    "per_seed": vals,
                }

    agg_recovery: dict = {}
    for stratum in ["overall", "single_hop", "multi_hop"]:
        agg_recovery[stratum] = {}
        for rm in repair_modes:
            rates   = [s["recovery"][stratum][rm]["rate"] for s in per_seed]
            brokens = [s["recovery"][stratum][rm]["corrupted_broken"] for s in per_seed]
            fixeds  = [s["recovery"][stratum][rm]["repaired_fixed"] for s in per_seed]
            agg_recovery[stratum][rm] = {
                "rate":             {"mean": _mean(rates),   "std": _std(rates),   "per_seed": rates},
                "corrupted_broken": {"mean": _mean(brokens), "per_seed": brokens},
                "repaired_fixed":   {"mean": _mean(fixeds),  "per_seed": fixeds},
            }

    return {"modes": modes, "metrics": agg_metrics, "recovery": agg_recovery}


# ---------------------------------------------------------------------------
# Printing — single-seed
# ---------------------------------------------------------------------------

def _fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) and v == v else "nan"


def print_single_table(title: str, stats: dict, stratum: str, n: int) -> None:
    cols = METRIC_KEYS + ["em_given_full_retrieval"]
    col_labels = ["EM", "Contains", "Recall@K", "AllSupport@K", "EM|FullRetr"]
    header = f"{'Condition':<22} {'n':>5}  " + "  ".join(f"{c:>13}" for c in col_labels)
    print(f"\n### {title}")
    print(header)
    print("-" * len(header))
    for mode in stats["modes"]:
        m = stats["metrics"][stratum][mode]
        vals = "  ".join(f"{_fmt(m.get(k, float('nan'))):>13}" for k in cols)
        print(f"{mode:<22} {n:>5}  {vals}")


def print_single_recovery(stats: dict) -> None:
    repair_modes = [m for m in stats["modes"] if m not in ("baseline", "corrupted")]
    for rm in repair_modes:
        label_suffix = " (iterative)" if rm == "repaired_iterative" else " (single-shot)"
        print(f"\n### Repair Recovery Rate{label_suffix}  (EM: corrupted wrong → {rm} right)")
        print(f"{'Stratum':<25} {'Broken':>8} {'Fixed':>8} {'Rate':>8}")
        print("-" * 55)
        for key, label in [
            ("overall", "overall"),
            ("single_hop", "single-hop-sufficient"),
            ("multi_hop", "genuinely-multi-hop"),
        ]:
            rec = stats["recovery"][key][rm]
            print(f"{label:<25} {rec['corrupted_broken']:>8} {rec['repaired_fixed']:>8} {rec['rate']:>8.3f}")


def print_single_seed(stats: dict) -> None:
    n = stats["n"]

    print(f"\n## Stratification")
    print(f"  Total:                    {n['total']}")
    print(f"  answer-not-in-support:    {n['excluded']}  (excluded)")
    print(f"  single-hop-sufficient:    {n['single_hop']}")
    print(f"  genuinely-multi-hop:      {n['multi_hop']}")
    print(f"    of which yes/no:        {n['yesno']}")
    print(f"    of which non-yes/no:    {n['bridge']}")

    print_single_table("Overall (excl. answer-not-in-support)", stats, "overall",   n["single_hop"] + n["multi_hop"])
    print_single_table("Single-hop-sufficient",                 stats, "single_hop", n["single_hop"])
    print_single_table("Genuinely-multi-hop",                   stats, "multi_hop",  n["multi_hop"])
    print_single_table("comparison-yesno",                      stats, "yesno",      n["yesno"])
    print_single_table("bridge-nonyesno",                       stats, "bridge",     n["bridge"])
    print_single_recovery(stats)


# ---------------------------------------------------------------------------
# Printing — multi-seed (mean ± std)
# ---------------------------------------------------------------------------

_CW = 17  # cell width for "0.XXX ± 0.XXX"


def _fmt_agg(entry: dict) -> str:
    mu, sd = entry["mean"], entry["std"]
    if mu != mu:
        return "nan"
    return f"{mu:.3f} ± {sd:.3f}"


def print_agg_table(title: str, stratum: str, agg: dict, n_label: str) -> None:
    modes = agg["modes"]
    cols = METRIC_KEYS + ["em_given_full_retrieval"]
    col_labels = ["EM", "Contains", "Recall@K", "AllSupport@K", "EM|FullRetr"]
    header = f"{'Condition':<22}  " + "  ".join(f"{c:^{_CW}}" for c in col_labels)
    print(f"\n### {title}  ({n_label})")
    print(header)
    print("-" * len(header))
    for mode in modes:
        cells = []
        for k in cols:
            entry = agg["metrics"][stratum][mode].get(k, {"mean": float("nan"), "std": 0.0})
            cells.append(f"{_fmt_agg(entry):^{_CW}}")
        print(f"{mode:<22}  " + "  ".join(cells))


def print_agg_recovery(agg: dict, n_seeds: int) -> None:
    repair_modes = [m for m in agg["modes"] if m not in ("baseline", "corrupted")]
    for rm in repair_modes:
        label_suffix = " (iterative)" if rm == "repaired_iterative" else " (single-shot)"
        print(f"\n### Repair Recovery Rate{label_suffix}  (mean ± std, {n_seeds} seeds)")
        print(f"{'Stratum':<25} {'Broken (mean)':>14} {'Fixed (mean)':>13} {'Rate':>17}")
        print("-" * 73)
        for key, label in [
            ("overall", "overall"),
            ("single_hop", "single-hop-sufficient"),
            ("multi_hop", "genuinely-multi-hop"),
        ]:
            rec = agg["recovery"][key][rm]
            broken_mean = rec["corrupted_broken"]["mean"]
            fixed_mean  = rec["repaired_fixed"]["mean"]
            rate_str    = _fmt_agg(rec["rate"])
            print(f"{label:<25} {broken_mean:>14.1f} {fixed_mean:>13.1f} {rate_str:>17}")


def print_multi_seed(agg: dict, per_seed_n: list[dict]) -> None:
    def n_label(stratum: str) -> str:
        if stratum == "overall":
            vals = [s["n"]["single_hop"] + s["n"]["multi_hop"] for s in per_seed_n]
        else:
            vals = [s["n"][stratum] for s in per_seed_n]
        lo, hi = min(vals), max(vals)
        return f"n={lo}" if lo == hi else f"n={lo}–{hi}"

    # aggregate n for stratification printout
    def n_mean(key):
        return _mean([s["n"][key] for s in per_seed_n])

    print(f"\n## Stratification (mean across {len(per_seed_n)} seeds)")
    print(f"  Total/seed:               {n_mean('total'):.0f}")
    print(f"  answer-not-in-support:    {n_mean('excluded'):.0f}  (excluded)")
    print(f"  single-hop-sufficient:    {n_mean('single_hop'):.0f}")
    print(f"  genuinely-multi-hop:      {n_mean('multi_hop'):.0f}")
    print(f"    of which yes/no:        {n_mean('yesno'):.0f}")
    print(f"    of which non-yes/no:    {n_mean('bridge'):.0f}")

    print_agg_table("Overall (excl. answer-not-in-support)", "overall",   agg, n_label("overall"))
    print_agg_table("Single-hop-sufficient",                 "single_hop", agg, n_label("single_hop"))
    print_agg_table("Genuinely-multi-hop",                   "multi_hop",  agg, n_label("multi_hop"))
    print_agg_table("comparison-yesno",                      "yesno",      agg, n_label("yesno"))
    print_agg_table("bridge-nonyesno",                       "bridge",     agg, n_label("bridge"))
    print_agg_recovery(agg, len(per_seed_n))


# ---------------------------------------------------------------------------
# Partial-to-full retrieval lift (iterative vs single-shot)
# ---------------------------------------------------------------------------

def _print_lift(rows: list[dict]) -> None:
    """Of examples where single-shot repair got partial (Recall@K=1, AllSupport@K=0),
    how many did iterative repair lift to AllSupport@K=1?"""
    partial = [
        r for r in rows
        if r["repaired"]["metrics"]["recall_at_k"] == 1
        and r["repaired"]["metrics"]["all_support_recall_at_k"] == 0
    ]
    lifted = [
        r for r in partial
        if r["repaired_iterative"]["metrics"]["all_support_recall_at_k"] == 1
    ]
    also_lifted_em = [r for r in lifted if r["repaired_iterative"]["metrics"]["exact_match"] == 1]

    print("\n### Partial-to-Full Retrieval Lift (iterative repair hypothesis test)")
    print(f"  Single-shot partial support (Recall@K=1, AllSupport@K=0): {len(partial)}")
    print(f"  Iterative lifted to AllSupport@K=1:                        {len(lifted)}"
          f"  ({len(lifted)/len(partial):.1%})" if partial else "  (n=0)")
    print(f"  Of those lifted, also gained EM=1:                         {len(also_lifted_em)}"
          f"  ({len(also_lifted_em)/len(lifted):.1%})" if lifted else "  (n=0)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=Path, default=DEFAULT_RESULTS_PATH,
                        help="Single result file (single-seed mode)")
    parser.add_argument("--inputs", type=str, default=None,
                        help="Comma-separated result files (multi-seed mode)")
    args = parser.parse_args()

    corpus = {doc["id"]: doc["text"] for doc in load_jsonl(CORPUS_PATH)}

    if args.inputs:
        paths = [Path(p.strip()) for p in args.inputs.split(",")]
        per_seed_stats = []
        for p in paths:
            rows = load_jsonl(p)
            modes = [m for m in _ALL_MODES if m in rows[0]]
            per_seed_stats.append(compute_seed_stats(rows, corpus, modes))
            print(f"  Loaded {len(rows)} rows from {p.name}")

        agg = aggregate_seeds(per_seed_stats)
        print_multi_seed(agg, per_seed_stats)

        output = {"n_seeds": len(paths), "per_seed": per_seed_stats, "aggregated": agg}
        MULTI_OUTPUT_PATH.write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {MULTI_OUTPUT_PATH}")

    else:
        rows = load_jsonl(args.input)
        modes = [m for m in _ALL_MODES if m in rows[0]]
        stats = compute_seed_stats(rows, corpus, modes)
        print_single_seed(stats)

        if "repaired_iterative" in modes:
            _print_lift(rows)

        OUTPUT_PATH.write_text(json.dumps(stats, indent=2))
        print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
