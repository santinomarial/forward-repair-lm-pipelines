"""Read-only replay of the final natural-error holdout, without provider calls."""

import json

import streamlit as st

from config import NATURAL_STUDY_DIR
from data_loader import load_jsonl
from natural_analysis import verify_evidence
from natural_routing import NATURAL_ACTIONS, NaturalRouter
from natural_study import model_lock, validate_rows, verify_inputs
from reliability_study import digest
from replay import answer_card


PRIMARY_POLICY = "natural_damage_augmented"


@st.cache_data(show_spinner="Verifying natural-error results…")
def snapshot(fingerprint: tuple) -> tuple:
    directory = NATURAL_STUDY_DIR
    manifest = verify_inputs(directory)
    if json.loads((directory / "test_model_lock.json").read_text()) != model_lock(directory):
        raise ValueError("frozen model or analysis changed")
    summary = json.loads((directory / "summary.json").read_text())
    for field, filename in (("manifest_sha256", "manifest.json"), ("models_sha256", "models.json"),
                            ("test_sha256", "test_outcomes.jsonl")):
        if summary[field] != digest(directory / filename):
            raise ValueError("summary provenance mismatch")
    rows = load_jsonl(directory / "test_outcomes.jsonl")
    validate_rows(directory, rows, ("test",))
    if len(rows) != len(manifest["splits"]["test"]):
        raise ValueError("incomplete held-out observations")
    verify_evidence(directory, rows)
    models = json.loads((directory / "models.json").read_text())
    policy = NaturalRouter(**models["policies"][PRIMARY_POLICY]["policy"])
    corpus = {d["id"]: d for d in load_jsonl(directory / "corpus.jsonl")}
    for row in rows:
        row["selected_action"] = policy.decide(row)
        before = row["initial"]["metrics"]["exact_match"]
        after = row["outcomes"][row["selected_action"]]["metrics"]["exact_match"]
        row["transition"] = {(0, 1): "Recovered", (1, 0): "Harmed", (1, 1): "Still exact match",
                             (0, 0): "Still not exact match"}[(before, after)]
    return summary, sorted(rows, key=lambda r: r["question_id"]), corpus


def render() -> None:
    st.caption("NATURAL ERRORS / FRESH HOLDOUT / OFFLINE REPLAY")
    st.title("Does repair help without injected failures?")
    st.write("Ordinary pipeline outputs, frozen decisions, and both recovery and damage. "
             "These are saved benchmark outcomes—not live generations or production traffic.")
    try:
        names = ("manifest.json", "models.json", "synthetic_model.json", "test_model_lock.json", "summary.json",
                 "development_outcomes.jsonl", "test_outcomes.jsonl", "examples.jsonl", "corpus.jsonl")
        paths = [NATURAL_STUDY_DIR / name for name in names]
        report, rows, corpus = snapshot(tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.info(f"The completed natural-error study is not available or could not be verified: {exc}")
        st.caption("No API call was attempted. The original saved experiments remain available.")
        return
    a, b, c, d = st.columns(4)
    a.metric("Held-out questions", len(rows))
    b.metric("Primary-policy recoveries", sum(r["transition"] == "Recovered" for r in rows))
    c.metric("Primary-policy regressions", sum(r["transition"] == "Harmed" for r in rows))
    d.metric("New API calls", 0)
    effect = report["comparisons"][report["primary_comparison"]]["exact_match"]
    st.info(f"Damage-aware router versus no repair: {100*effect['estimate']:+.1f} pp EM "
            f"[95% CI {100*effect['ci'][0]:+.1f}, {100*effect['ci'][1]:+.1f}]. "
            "Exact match is lexical; human semantic review is still pending.")
    with st.expander("All policies · held-out results"):
        st.dataframe([{"Policy": name, "EM (%)": round(100*p["exact_match"]["estimate"], 1),
                       "Recovered": p["recovered_count"], "Damaged": p["damaged_count"],
                       "Added calls": round(p["incremental"]["calls_per_question"], 2),
                       "Added USD/question": f"{p['incremental']['cost_usd_per_question']['estimate']:.6f}"}
                      for name, p in report["policies"].items()], hide_index=True, width="stretch")
    with st.sidebar:
        st.divider()
        view = st.selectbox("Natural output", ["All", "Recovered", "Harmed", "Still exact match", "Still not exact match"])
        search = st.text_input("Find natural question")
    selected_rows = [r for r in rows if (view == "All" or r["transition"] == view)
                     and search.casefold() in (r["question"] + r["question_id"]).casefold()]
    if not selected_rows:
        st.info("No natural cases match these filters.")
        return
    by_id = {r["question_id"]: r for r in selected_rows}
    ids = list(by_id)
    linked = st.query_params.get("natural_case")
    selected = st.selectbox(f"Natural case · {len(ids)} matching", ids,
                            index=ids.index(linked) if linked in ids else 0,
                            format_func=lambda key: by_id[key]["question"], key="natural_case")
    st.query_params["natural_case"] = selected
    row = by_id[selected]
    st.subheader(row["question"])
    st.caption(f"Reference: {row['gold']} · Primary router: {row['selected_action']} · {row['transition']}")
    action = st.selectbox("Saved natural action", NATURAL_ACTIONS,
                          index=NATURAL_ACTIONS.index(row["selected_action"]), key=f"natural_action_{selected}")
    if action != row["selected_action"]:
        st.caption("Saved alternative—not the primary router's decision.")
    outcome = row["outcomes"][action]
    usage = outcome["telemetry"]
    st.caption(f"Added calls: {usage['llm_calls']} · Tokens: {usage['total_tokens']:,} · "
               f"Estimated cost: ${usage['estimated_cost_usd']:.6f} · Wall time: {usage['wall_clock_seconds']:.2f}s "
               "(includes pacing; excludes original pass and router CPU). Replay is free.")
    left, right = st.columns(2)
    with left:
        answer_card("Before · natural output", row["initial"]["answer"], row["initial"]["query"],
                    row["initial"]["metrics"], row["initial"]["docs"], row["support_doc_ids"])
    with right:
        answer_card(f"After · {action}", outcome["answer"], outcome["query"], outcome["metrics"],
                    [corpus[key] for key in outcome["doc_ids"]], row["support_doc_ids"])
    st.caption("Benchmark support markers are post-hoc annotations, never router features. "
               "All six actions were observed during collection; the deployed-style decision used only the initial state.")
    export = {"case": row, "documents": {key: corpus[key] for key in sorted(
        {key for o in row["outcomes"].values() for key in o["doc_ids"]})},
        "provenance": {key: report[key] for key in ("manifest_sha256", "models_sha256", "test_sha256")}}
    st.download_button("Download natural case + evidence", json.dumps(export, indent=2),
                       f"natural_case_{selected}.json", mime="application/json")
