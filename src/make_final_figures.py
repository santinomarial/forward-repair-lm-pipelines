"""Generate all final figures and tables for the forward repair report."""
import json
import math
import re
import string
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from config import (
    CORPUS_PATH,
    FIGURE_ANSWER_PATH,
    FIGURE_ITERATIVE_PATH,
    FIGURE_QUERY_SEED_PATHS,
    FIGURES_DIR,
    TABLES_DIR,
)

OUT_FIGS = FIGURES_DIR
OUT_TABS = TABLES_DIR
QUERY_SEEDS = list(FIGURE_QUERY_SEED_PATHS)
ANSWER_FILE = FIGURE_ANSWER_PATH
ITERATIVE_FILE = FIGURE_ITERATIVE_PATH
CORPUS_FILE = CORPUS_PATH

# Wong colorblind-safe palette
COLORS = {
    "baseline":         "#0072B2",
    "corrupted_query":  "#D55E00",
    "repaired_single":  "#009E73",
    "repaired_iterative": "#E69F00",
    "corrupted_answer": "#CC79A7",
    "repaired_answer":  "#56B4E9",
}

METRIC_LABELS = {
    "exact_match":            "Exact Match",
    "contains_answer":        "Contains Answer",
    "recall_at_k":            "Recall@K",
    "all_support_recall_at_k": "AllSupport@K",
}


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def ensure_files() -> None:
    missing = [f for f in [*QUERY_SEEDS, ANSWER_FILE, ITERATIVE_FILE, CORPUS_FILE] if not f.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files:\n" + "\n".join(str(f) for f in missing))


# ---------------------------------------------------------------------------
# Hop classification
# ---------------------------------------------------------------------------

def _norm_classify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify(gold: str, support_doc_ids: list[str], corpus: dict[str, str]) -> str:
    if gold.lower().strip() in ("yes", "no"):
        return "genuinely-multi-hop"
    norm_gold = _norm_classify(gold)
    matches = sum(1 for doc_id in support_doc_ids if norm_gold in _norm_classify(corpus.get(doc_id, "")))
    if matches == 0:
        return "answer-not-in-support"
    if matches == 1:
        return "single-hop-sufficient"
    return "genuinely-multi-hop"


def get_strata(rows: list[dict], corpus: dict[str, str]) -> dict[str, list[dict]]:
    classes = [classify(r["gold"], r["support_doc_ids"], corpus) for r in rows]
    single = [r for r, c in zip(rows, classes) if c == "single-hop-sufficient"]
    multi  = [r for r, c in zip(rows, classes) if c == "genuinely-multi-hop"]
    yesno  = [r for r in multi if r["gold"].lower().strip() in ("yes", "no")]
    bridge = [r for r in multi if r["gold"].lower().strip() not in ("yes", "no")]
    return {
        "overall":    single + multi,
        "single_hop": single,
        "multi_hop":  multi,
        "yesno":      yesno,
        "bridge":     bridge,
    }


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float("nan")


def _std(vals: list[float]) -> float:
    n = len(vals)
    if n < 2:
        return 0.0
    mu = _mean(vals)
    return math.sqrt(sum((v - mu) ** 2 for v in vals) / (n - 1))


def compute_metrics(rows: list[dict], mode: str) -> dict:
    keys = ["exact_match", "contains_answer", "recall_at_k", "all_support_recall_at_k"]
    result = {k: _mean([r[mode]["metrics"][k] for r in rows]) for k in keys}
    em_full = [r[mode]["metrics"]["exact_match"] for r in rows
               if r[mode]["metrics"]["all_support_recall_at_k"] == 1]
    result["em_given_full_retrieval"] = _mean(em_full)
    return result


def recovery_rate(rows: list[dict], repair_mode: str = "repaired") -> dict:
    broken = [r for r in rows if r["corrupted"]["metrics"]["exact_match"] == 0]
    fixed  = [r for r in broken if r[repair_mode]["metrics"]["exact_match"] == 1]
    return {
        "broken": len(broken),
        "fixed":  len(fixed),
        "rate":   len(fixed) / len(broken) if broken else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-seed stats for mean ± std
# ---------------------------------------------------------------------------

def seed_stats_for(seed_rows_list: list[list[dict]], corpus: dict[str, str]) -> dict:
    """Return {stratum: {mode: {metric: {'mean':, 'std':, 'per_seed':}}}}."""
    per_seed = []
    for rows in seed_rows_list:
        strata = get_strata(rows, corpus)
        modes  = [m for m in ["baseline", "corrupted", "repaired", "repaired_iterative"] if m in rows[0]]
        entry  = {}
        for stratum, group in strata.items():
            entry[stratum] = {mode: compute_metrics(group, mode) for mode in modes if group}
        # recovery
        entry["_recovery"] = {}
        for stratum, group in [("overall", strata["overall"]),
                               ("single_hop", strata["single_hop"]),
                               ("multi_hop", strata["multi_hop"])]:
            entry["_recovery"][stratum] = {}
            for rm in [m for m in modes if m not in ("baseline", "corrupted")]:
                entry["_recovery"][stratum][rm] = recovery_rate(group, rm)
        per_seed.append(entry)

    strata_keys = list(per_seed[0].keys())
    modes_present = [m for m in ["baseline", "corrupted", "repaired", "repaired_iterative"]
                     if m in per_seed[0].get("overall", {})]
    metric_keys = ["exact_match", "contains_answer", "recall_at_k",
                   "all_support_recall_at_k", "em_given_full_retrieval"]

    agg = {}
    for stratum in [k for k in strata_keys if not k.startswith("_")]:
        agg[stratum] = {}
        for mode in modes_present:
            agg[stratum][mode] = {}
            for k in metric_keys:
                vals = [s[stratum][mode].get(k, float("nan")) for s in per_seed
                        if stratum in s and mode in s[stratum]]
                valid = [v for v in vals if v == v]
                agg[stratum][mode][k] = {"mean": _mean(valid), "std": _std(valid), "per_seed": valid}

    agg["_recovery"] = {}
    repair_modes = [m for m in modes_present if m not in ("baseline", "corrupted")]
    for stratum in ["overall", "single_hop", "multi_hop"]:
        agg["_recovery"][stratum] = {}
        for rm in repair_modes:
            rates   = [s["_recovery"][stratum][rm]["rate"] for s in per_seed]
            brokens = [s["_recovery"][stratum][rm]["broken"] for s in per_seed]
            fixeds  = [s["_recovery"][stratum][rm]["fixed"] for s in per_seed]
            agg["_recovery"][stratum][rm] = {
                "rate":   {"mean": _mean(rates),   "std": _std(rates),   "per_seed": rates},
                "broken": {"mean": _mean(brokens), "per_seed": brokens},
                "fixed":  {"mean": _mean(fixeds),  "per_seed": fixeds},
            }

    return agg


# ---------------------------------------------------------------------------
# Figure 1: Main results — grouped bar chart with error bars
# ---------------------------------------------------------------------------

def fig_main_results(query_agg: dict, iterative_rows: list[dict], corpus: dict[str, str]) -> None:
    iter_strata = get_strata(iterative_rows, corpus)
    iter_overall = iter_strata["overall"]

    metrics = ["exact_match", "contains_answer", "recall_at_k", "all_support_recall_at_k"]
    labels  = [METRIC_LABELS[m] for m in metrics]

    conditions = [
        ("Baseline",           "baseline",           COLORS["baseline"],           False),
        ("Corrupted (query)",  "corrupted",           COLORS["corrupted_query"],    True),
        ("Repaired (single)",  "repaired",            COLORS["repaired_single"],    True),
        ("Repaired (iter.)",   "repaired_iterative",  COLORS["repaired_iterative"], False),
    ]

    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-(len(conditions) - 1) * width / 2,
                           (len(conditions) - 1) * width / 2, len(conditions))

    fig, ax = plt.subplots(figsize=(10, 5))

    for (label, mode, color, has_error), offset in zip(conditions, offsets):
        if mode == "repaired_iterative":
            vals = [compute_metrics(iter_overall, mode)[m] for m in metrics]
            errs = [0.0] * len(metrics)
        else:
            vals = [query_agg["overall"][mode][m]["mean"] for m in metrics]
            errs = [query_agg["overall"][mode][m]["std"] for m in metrics] if has_error else [0.0] * len(metrics)

        bars = ax.bar(x + offset, vals, width, label=label, color=color,
                      yerr=errs if any(e > 0 for e in errs) else None,
                      capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "black"})

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.set_title("Forward Repair — Main Results (n=300)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_FIGS / "main_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 2: Asymmetry — query vs answer corruption
# ---------------------------------------------------------------------------

def fig_asymmetry(query_agg: dict, answer_rows: list[dict], corpus: dict[str, str]) -> None:
    answer_strata = get_strata(answer_rows, corpus)
    answer_overall = answer_strata["overall"]

    # EM values
    base_em = query_agg["overall"]["baseline"]["exact_match"]["mean"]

    q_corr_em = query_agg["overall"]["corrupted"]["exact_match"]["mean"]
    q_corr_err = query_agg["overall"]["corrupted"]["exact_match"]["std"]
    q_rep_em   = query_agg["overall"]["repaired"]["exact_match"]["mean"]
    q_rep_err  = query_agg["overall"]["repaired"]["exact_match"]["std"]

    a_corr_em = compute_metrics(answer_overall, "corrupted")["exact_match"]
    a_rep_em  = compute_metrics(answer_overall, "repaired")["exact_match"]

    q_rec = recovery_rate(
        [r for rows in [load_jsonl(f) for f in QUERY_SEEDS] for r in get_strata(rows, corpus)["overall"]],
        "repaired"
    )
    a_rec = recovery_rate(answer_overall, "repaired")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Left: EM comparison
    ax = axes[0]
    categories = ["Baseline", "Corrupted", "Repaired"]
    q_vals = [base_em, q_corr_em, q_rep_em]
    q_errs = [0.0, q_corr_err, q_rep_err]
    a_vals = [base_em, a_corr_em, a_rep_em]

    x = np.arange(len(categories))
    w = 0.32
    ax.bar(x - w / 2, q_vals, w, label="Query corruption",  color=COLORS["corrupted_query"],
           yerr=q_errs, capsize=3, error_kw={"elinewidth": 1.2, "ecolor": "black"}, alpha=0.9)
    ax.bar(x + w / 2, a_vals, w, label="Answer corruption", color=COLORS["corrupted_answer"], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylabel("Exact Match", fontsize=12)
    ax.set_ylim(0, 0.55)
    ax.set_title("EM by Corruption Stage", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Right: Recovery rate comparison
    ax2 = axes[1]
    stages = ["Query corruption", "Answer corruption"]
    rates  = [q_rec["rate"], a_rec["rate"]]
    colors = [COLORS["corrupted_query"], COLORS["corrupted_answer"]]
    bars = ax2.bar(stages, rates, color=colors, width=0.45, alpha=0.9)
    for bar, rate in zip(bars, rates):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{rate:.1%}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Recovery Rate", fontsize=12)
    ax2.set_ylim(0, 0.38)
    ax2.set_title("Recovery Rate: Query vs Answer", fontsize=12, fontweight="bold")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax2.set_axisbelow(True)

    plt.suptitle("Repair Asymmetry: Query-Stage vs Answer-Stage Corruption",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = OUT_FIGS / "asymmetry_query_vs_answer.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 3: Hop stratification
# ---------------------------------------------------------------------------

def fig_hop_stratification(query_agg: dict, iterative_rows: list[dict], corpus: dict[str, str]) -> None:
    iter_strata = get_strata(iterative_rows, corpus)
    strata_names = ["single_hop", "multi_hop"]
    strata_labels = ["Single-hop-sufficient", "Genuinely-multi-hop"]

    conditions = [
        ("Baseline",          "baseline",          COLORS["baseline"],           False),
        ("Corrupted (query)", "corrupted",          COLORS["corrupted_query"],    True),
        ("Repaired (single)", "repaired",           COLORS["repaired_single"],    True),
        ("Repaired (iter.)",  "repaired_iterative", COLORS["repaired_iterative"], False),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, stratum, title in zip(axes, strata_names, strata_labels):
        x = np.arange(len(conditions))
        vals, errs, colors, lbls = [], [], [], []
        for (label, mode, color, has_error) in conditions:
            if mode == "repaired_iterative":
                v = compute_metrics(iter_strata[stratum], mode)["exact_match"]
                e = 0.0
            else:
                v = query_agg[stratum][mode]["exact_match"]["mean"]
                e = query_agg[stratum][mode]["exact_match"]["std"] if has_error else 0.0
            vals.append(v)
            errs.append(e)
            colors.append(color)
            lbls.append(label)

        bars = ax.bar(x, vals, color=colors, width=0.55)
        for i, (v, e) in enumerate(zip(vals, errs)):
            if e > 0:
                ax.errorbar(x[i], v, yerr=e, fmt="none", color="black",
                            capsize=3, elinewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(lbls, fontsize=9, rotation=15, ha="right")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Exact Match", fontsize=11)
        ax.set_ylim(0, 0.85)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        ax.set_axisbelow(True)

    plt.suptitle("Exact Match by Hop Complexity", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = OUT_FIGS / "hop_stratification.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 4: Partial-to-full retrieval lift (stepped bar)
# ---------------------------------------------------------------------------

def fig_partial_to_full_lift(iterative_rows: list[dict]) -> None:
    partial = [
        r for r in iterative_rows
        if r["repaired"]["metrics"]["recall_at_k"] == 1
        and r["repaired"]["metrics"]["all_support_recall_at_k"] == 0
    ]
    lifted     = [r for r in partial if r["repaired_iterative"]["metrics"]["all_support_recall_at_k"] == 1]
    em_gained  = [r for r in lifted  if r["repaired_iterative"]["metrics"]["exact_match"] == 1]

    not_lifted  = [r for r in partial if r not in lifted]
    lifted_no_em = [r for r in lifted  if r not in em_gained]

    n_partial   = len(partial)
    n_lifted    = len(lifted)
    n_em        = len(em_gained)
    n_not_lifted = len(not_lifted)
    n_lifted_no_em = len(lifted_no_em)

    fig, ax = plt.subplots(figsize=(9, 5))

    categories = [
        f"Single-shot partial\n(n={n_partial})",
        f"Iterative lifts AllSupport\n(n={n_lifted})",
        f"Also gains EM\n(n={n_em})",
    ]
    vals = [n_partial, n_lifted, n_em]
    pcts = [100.0, 100.0 * n_lifted / n_partial, 100.0 * n_em / n_partial]

    bar_colors = [COLORS["repaired_single"], COLORS["repaired_iterative"], COLORS["baseline"]]
    x = np.arange(len(categories))
    bars = ax.bar(x, vals, color=bar_colors, width=0.55, alpha=0.9)

    for bar_obj, v, p in zip(bars, vals, pcts):
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, v + 1.5,
                f"{v}\n({p:.0f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylabel("Number of Examples", fontsize=12)
    ax.set_ylim(0, n_partial * 1.25)
    ax.set_title("Partial-to-Full Retrieval Lift via Iterative Repair\n"
                 "(examples with Recall@K=1 but AllSupport@K=0 under single-shot repair)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_FIGS / "partial_to_full_lift.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Figure 5: Recovery rate by stratum (single-shot vs iterative)
# ---------------------------------------------------------------------------

def fig_recovery_by_stratum(iterative_rows: list[dict], query_agg: dict, corpus: dict[str, str]) -> None:
    strata_def = get_strata(iterative_rows, corpus)

    stratum_labels = [
        ("overall",    "Overall"),
        ("single_hop", "Single-hop"),
        ("multi_hop",  "Multi-hop"),
        ("yesno",      "Yes/No (multi)"),
        ("bridge",     "Bridge (multi)"),
    ]

    single_rates, iter_rates = [], []
    for key, _ in stratum_labels:
        group = strata_def[key]
        single_rates.append(recovery_rate(group, "repaired")["rate"])
        iter_rates.append(recovery_rate(group, "repaired_iterative")["rate"])

    y = np.arange(len(stratum_labels))
    h = 0.32
    labels = [lab for _, lab in stratum_labels]

    fig, ax = plt.subplots(figsize=(9, 5))
    b1 = ax.barh(y + h / 2, single_rates, h, label="Single-shot repair",
                 color=COLORS["repaired_single"], alpha=0.9)
    b2 = ax.barh(y - h / 2, iter_rates,   h, label="Iterative repair",
                 color=COLORS["repaired_iterative"], alpha=0.9)

    for bars in [b1, b2]:
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1%}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Recovery Rate", fontsize=12)
    ax.set_xlim(0, 0.75)
    ax.set_title("Recovery Rate by Stratum\n(corrupted-wrong → repaired-right, EM)",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = OUT_FIGS / "recovery_by_stratum.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fmt(v, std=None) -> str:
    if v != v:
        return "nan"
    s = f"{v:.3f}"
    if std is not None and std > 0:
        s += f" ± {std:.3f}"
    return s


def table_main(query_agg: dict, iterative_rows: list[dict], corpus: dict[str, str],
               answer_rows: list[dict]) -> None:
    iter_strata = get_strata(iterative_rows, corpus)
    ans_strata  = get_strata(answer_rows, corpus)

    metrics = ["exact_match", "contains_answer", "recall_at_k", "all_support_recall_at_k"]
    header  = ["Condition"] + [METRIC_LABELS[m] for m in metrics] + ["Recovery Rate"]

    rows_data = []
    base_em = query_agg["overall"]["baseline"]["exact_match"]["mean"]

    def q_rec_agg():
        rates = query_agg["_recovery"]["overall"]["repaired"]["rate"]
        return rates["mean"], rates["std"]

    # Baseline
    row = ["Baseline (query)"]
    for m in metrics:
        row.append(_fmt(query_agg["overall"]["baseline"][m]["mean"]))
    row.append("—")
    rows_data.append(row)

    # Corrupted query
    row = ["Corrupted (query)"]
    for m in metrics:
        v = query_agg["overall"]["corrupted"][m]
        row.append(_fmt(v["mean"], v["std"]))
    row.append("—")
    rows_data.append(row)

    # Repaired single
    mu, sd = q_rec_agg()
    row = ["Repaired single-shot (query)"]
    for m in metrics:
        v = query_agg["overall"]["repaired"][m]
        row.append(_fmt(v["mean"], v["std"]))
    row.append(_fmt(mu, sd))
    rows_data.append(row)

    # Repaired iterative
    iter_met = compute_metrics(iter_strata["overall"], "repaired_iterative")
    iter_rec = recovery_rate(iter_strata["overall"], "repaired_iterative")
    row = ["Repaired iterative (query)"]
    for m in metrics:
        row.append(_fmt(iter_met[m]))
    row.append(_fmt(iter_rec["rate"]))
    rows_data.append(row)

    # Blank separator
    rows_data.append([""] * len(header))

    # Answer corruption
    ans_met_corr = compute_metrics(ans_strata["overall"], "corrupted")
    ans_met_rep  = compute_metrics(ans_strata["overall"], "repaired")
    ans_rec      = recovery_rate(ans_strata["overall"], "repaired")

    row = ["Corrupted (answer)"]
    for m in metrics:
        row.append(_fmt(ans_met_corr[m]))
    row.append("—")
    rows_data.append(row)

    row = ["Repaired (answer)"]
    for m in metrics:
        row.append(_fmt(ans_met_rep[m]))
    row.append(_fmt(ans_rec["rate"]))
    rows_data.append(row)

    lines = ["# Table 1: Main Results\n",
             "Error bars (± std) shown for conditions with 3 seeds (query corrupted, repaired single-shot).\n",
             "Iterative and answer corruption conditions use seed 0 only.\n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for row in rows_data:
        lines.append("| " + " | ".join(row) + " |")

    path = OUT_TABS / "main_table.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {path.name}")


def table_stratified(query_agg: dict, iterative_rows: list[dict], corpus: dict[str, str]) -> None:
    iter_strata = get_strata(iterative_rows, corpus)

    strata_info = [
        ("overall",    "Overall (excl. answer-not-in-support)"),
        ("single_hop", "Single-hop-sufficient"),
        ("multi_hop",  "Genuinely-multi-hop"),
        ("yesno",      "Yes/No (comparison)"),
        ("bridge",     "Bridge (non-yes/no)"),
    ]

    conditions = [
        ("Baseline",          "baseline",          False),
        ("Corrupted (query)", "corrupted",          True),
        ("Repaired (single)", "repaired",           True),
        ("Repaired (iter.)",  "repaired_iterative", False),
    ]

    metrics = ["exact_match", "recall_at_k", "all_support_recall_at_k"]
    metric_labels = ["EM", "Recall@K", "AllSupport@K"]
    header = ["Stratum", "Condition"] + metric_labels

    lines = ["# Table 2: Stratified Results — Query Corruption\n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for stratum_key, stratum_label in strata_info:
        first = True
        for cond_label, mode, has_error in conditions:
            row = [stratum_label if first else "", cond_label]
            first = False
            for m in metrics:
                if mode == "repaired_iterative":
                    v = compute_metrics(iter_strata[stratum_key], mode)[m]
                    row.append(_fmt(v))
                else:
                    v = query_agg[stratum_key][mode][m]
                    sd = v["std"] if has_error else None
                    row.append(_fmt(v["mean"], sd))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("| " + " | ".join([""] * len(header)) + " |")

    path = OUT_TABS / "stratified_table.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {path.name}")


def table_answer_corruption(answer_rows: list[dict], corpus: dict[str, str]) -> None:
    ans_strata = get_strata(answer_rows, corpus)

    strata_info = [
        ("overall",    "Overall"),
        ("single_hop", "Single-hop"),
        ("multi_hop",  "Multi-hop"),
        ("yesno",      "Yes/No"),
        ("bridge",     "Bridge"),
    ]

    metrics = ["exact_match", "recall_at_k", "all_support_recall_at_k"]
    metric_labels = ["EM", "Recall@K", "AllSupport@K"]
    header = ["Stratum", "Condition"] + metric_labels

    lines = ["# Table 3: Stratified Results — Answer Corruption\n"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for stratum_key, stratum_label in strata_info:
        first = True
        for cond_label, mode in [("Baseline", "baseline"),
                                  ("Corrupted (answer)", "corrupted"),
                                  ("Repaired (answer)", "repaired")]:
            row = [stratum_label if first else "", cond_label]
            first = False
            group = ans_strata[stratum_key]
            for m in metrics:
                v = compute_metrics(group, mode)[m]
                row.append(_fmt(v))
            lines.append("| " + " | ".join(row) + " |")
        lines.append("| " + " | ".join([""] * len(header)) + " |")

    path = OUT_TABS / "answer_corruption_table.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {path.name}")


def table_recovery(query_agg: dict, iterative_rows: list[dict], corpus: dict[str, str]) -> None:
    iter_strata = get_strata(iterative_rows, corpus)

    strata_info = [
        ("overall",    "Overall"),
        ("single_hop", "Single-hop"),
        ("multi_hop",  "Multi-hop"),
    ]

    lines = ["# Table 4: Recovery Rates by Stratum\n",
             "(corrupted EM=0 → repaired EM=1)\n"]
    header = ["Stratum", "Broken (mean)", "Fixed single (mean)", "Rate single",
              "Fixed iter. (mean)", "Rate iter.", "Gap (iter − single)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for stratum_key, stratum_label in strata_info:
        # single-shot from query_agg
        s_rate = query_agg["_recovery"][stratum_key]["repaired"]["rate"]
        s_broken = query_agg["_recovery"][stratum_key]["repaired"]["broken"]
        s_fixed  = query_agg["_recovery"][stratum_key]["repaired"]["fixed"]

        # iterative from data
        group = iter_strata[stratum_key]
        i_rec = recovery_rate(group, "repaired_iterative")
        s_rec_val = recovery_rate(group, "repaired")

        gap = i_rec["rate"] - s_rec_val["rate"]

        row = [
            stratum_label,
            f"{s_broken['mean']:.1f}",
            f"{s_fixed['mean']:.1f}",
            _fmt(s_rate["mean"], s_rate["std"]),
            f"{i_rec['fixed']}",
            _fmt(i_rec["rate"]),
            f"{gap:+.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")

    path = OUT_TABS / "recovery_table.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {path.name}")


def table_qualitative(iterative_rows: list[dict], corpus: dict[str, str]) -> None:
    strata = get_strata(iterative_rows, corpus)

    def pick_example(pool: list[dict], *, corrupted_wrong: bool, repaired_right: bool,
                     mode: str = "repaired") -> dict | None:
        for r in pool:
            c_ok = (r["corrupted"]["metrics"]["exact_match"] == 0) == corrupted_wrong
            r_ok = (r[mode]["metrics"]["exact_match"] == 1) == repaired_right
            if c_ok and r_ok:
                return r
        return None

    examples = []

    # 1. Single-hop recovered
    e = pick_example(strata["single_hop"], corrupted_wrong=True, repaired_right=True)
    if e:
        examples.append(("Single-hop: corrupted wrong, repaired right", e, "repaired"))

    # 2. Multi-hop recovered by iterative but not single-shot
    for r in strata["multi_hop"]:
        if (r["corrupted"]["metrics"]["exact_match"] == 0
                and r["repaired"]["metrics"]["exact_match"] == 0
                and r["repaired_iterative"]["metrics"]["exact_match"] == 1):
            examples.append(("Multi-hop: iterative repairs where single-shot failed", r, "repaired_iterative"))
            break

    # 3. Yes/No fully recovered
    e = pick_example(strata["yesno"], corrupted_wrong=True, repaired_right=True, mode="repaired_iterative")
    if e:
        examples.append(("Yes/No: iterative repair recovers", e, "repaired_iterative"))

    # 4. Single-shot partial retrieval (Recall@K=1, AllSupport@K=0) lifted by iterative
    for r in iterative_rows:
        if (r["repaired"]["metrics"]["recall_at_k"] == 1
                and r["repaired"]["metrics"]["all_support_recall_at_k"] == 0
                and r["repaired_iterative"]["metrics"]["all_support_recall_at_k"] == 1):
            examples.append(("Partial-to-full lift: iterative completes retrieval", r, "repaired_iterative"))
            break

    # 5. Failure: repair fails (still wrong)
    e = pick_example(strata["overall"], corrupted_wrong=True, repaired_right=False)
    if e:
        examples.append(("Repair failure: corrupted wrong, repaired still wrong", e, "repaired"))

    lines = ["# Table 5: Qualitative Examples\n"]
    for i, (title, row, highlight_mode) in enumerate(examples[:5], 1):
        lines.append(f"## Example {i}: {title}\n")
        lines.append(f"**Question:** {row['question']}\n")
        lines.append(f"**Gold answer:** {row['gold']}\n")
        lines.append(f"**Support docs:** {', '.join(row['support_doc_ids'])}\n")
        lines.append(f"\n**Baseline query:** {row['baseline'].get('query', '—')}")
        lines.append(f"**Baseline answer:** {row['baseline']['answer']}")
        lines.append(f"**Baseline EM:** {row['baseline']['metrics']['exact_match']}\n")
        lines.append(f"**Corrupted query:** {row['corrupted'].get('query', '—')}")
        lines.append(f"**Corrupted answer:** {row['corrupted']['answer']}")
        lines.append(f"**Corrupted EM:** {row['corrupted']['metrics']['exact_match']}\n")
        if highlight_mode == "repaired_iterative" and "repaired_iterative" in row:
            ri = row["repaired_iterative"]
            lines.append(f"**Repaired (single) answer:** {row['repaired']['answer']}")
            lines.append(f"**Repaired (single) EM:** {row['repaired']['metrics']['exact_match']}\n")
            lines.append(f"**Iterative reasoning:** {ri.get('reasoning', '—')}")
            lines.append(f"**Query A:** {ri.get('query_a', '—')}")
            lines.append(f"**Query B:** {ri.get('query_b', '—')}")
            lines.append(f"**Iterative answer:** {ri['answer']}")
            lines.append(f"**Iterative EM:** {ri['metrics']['exact_match']}\n")
        else:
            lines.append(f"**Repaired answer:** {row['repaired']['answer']}")
            lines.append(f"**Repaired EM:** {row['repaired']['metrics']['exact_match']}\n")
        lines.append("---\n")

    path = OUT_TABS / "qualitative_examples.md"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ensure_files()
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    OUT_TABS.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    corpus = {doc["id"]: doc["text"] for doc in load_jsonl(CORPUS_FILE)}

    query_seed_rows = [load_jsonl(f) for f in QUERY_SEEDS]
    answer_rows     = load_jsonl(ANSWER_FILE)
    iterative_rows  = load_jsonl(ITERATIVE_FILE)

    print("Computing multi-seed query stats...")
    query_agg = seed_stats_for(query_seed_rows, corpus)

    print("\nGenerating figures...")
    fig_main_results(query_agg, iterative_rows, corpus)
    fig_asymmetry(query_agg, answer_rows, corpus)
    fig_hop_stratification(query_agg, iterative_rows, corpus)
    fig_partial_to_full_lift(iterative_rows)
    fig_recovery_by_stratum(iterative_rows, query_agg, corpus)

    print("\nGenerating tables...")
    table_main(query_agg, iterative_rows, corpus, answer_rows)
    table_stratified(query_agg, iterative_rows, corpus)
    table_answer_corruption(answer_rows, corpus)
    table_recovery(query_agg, iterative_rows, corpus)
    table_qualitative(iterative_rows, corpus)

    print("\nDone.")
    print(f"  Figures → {OUT_FIGS}")
    print(f"  Tables  → {OUT_TABS}")


if __name__ == "__main__":
    main()
