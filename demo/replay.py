"""Streamlit presentation of verified saved outcomes and explicit reviewer input."""

import json

import streamlit as st

from case_explorer import (
    ACTION_LABELS, FAMILY_LABELS, Snapshot, case_export, evidence_for, filter_cases, load_snapshot,
)
from config import CORPUS_PATH, EXAMPLES_PATH, RELIABILITY_STUDY_DIR
from error_audit import REVIEW_LABELS, completed_review, review_sample, review_template
from outcome_routing import ACTIONS


@st.cache_data(show_spinner="Verifying frozen results and evidence…")
def saved_snapshot(fingerprint: tuple) -> Snapshot:
    return load_snapshot()


def documents(docs: list[dict], support_ids: list[str]) -> None:
    for rank, doc in enumerate(docs, 1):
        support = " · benchmark support" if doc["id"] in support_ids else ""
        with st.expander(f"{rank:02d} · {doc['title']}{support}"):
            st.caption(doc["id"])
            st.write(doc["text"])


def answer_card(title: str, answer: str, query: str, metrics: dict, docs: list[dict], support_ids: list[str]) -> None:
    with st.container(border=True):
        st.subheader(title)
        st.caption(f"Exact match: {'yes' if metrics['exact_match'] else 'no'} · Token F1: {metrics['token_f1']:.1%}")
        st.write(answer)
        st.markdown("**Search query**")
        st.code(query, language=None, wrap_lines=True)
        st.markdown(f"**Evidence · {len(docs)} documents**")
        documents(docs, support_ids)


def review_panel(snapshot: Snapshot, case: dict) -> None:
    st.caption("Post-hoc review signals—not factuality verdicts. Gold/support annotations are never router inputs.")
    if case["flags"]:
        for flag in case["flags"]:
            st.write(f"• {flag}")
    else:
        st.write("No automatic flags for the frozen policy's before/after pair.")
    st.info("A gold phrase can appear in a negated or incorrect answer. Missing annotated support may have "
            "alternative evidence. Read the passages before assigning a cause.")
    row = snapshot.rows[case["case_id"]]
    missing = sorted(set(case["missing_support_before"] + case["missing_support_after"]))
    if missing:
        with st.expander("Reference support absent from one or both runs"):
            st.warning("These are benchmark reference documents—not necessarily evidence the model received.")
            documents([snapshot.corpus[key] for key in missing], case["support_doc_ids"])
    st.markdown("**Annotate the frozen router's decision**")
    st.caption(f"Review target: {ACTION_LABELS[case['action']]}. Changing the comparison above does not change this target.")
    with st.expander("Label guide"):
        st.write({
            "format_only": "Correct answer expressed differently; EM fails.",
            "missing_evidence": "A necessary fact is absent from the supplied evidence.",
            "reasoning_error": "Evidence is sufficient but the conclusion is wrong.",
            "unsupported_claim": "A material claim is not supported by supplied evidence.",
            "reference_ambiguity": "Question/reference or accepted-answer scope is ambiguous.",
            "no_error": "Correct, evidence-supported answer.", "uncertain": "Cannot establish a verdict.",
        })
    allowed_ids = sorted({doc["id"] for doc in row["initial"]["docs"]}
                         | set(row["outcomes"][case["action"]]["doc_ids"]) | set(case["support_doc_ids"]))
    with st.form(f"review_{case['case_id']}"):
        reviewer = st.text_input("Reviewer identifier", help="Use an alias if you plan to share the exported file.")
        before = st.multiselect("Initial answer labels", REVIEW_LABELS)
        after = st.multiselect("Selected answer labels", REVIEW_LABELS)
        evidence = st.multiselect("Evidence document IDs", allowed_ids)
        rationale = st.text_area("Evidence-based rationale", placeholder="Cite the passages and explain the before/after judgment.")
        submitted = st.form_submit_button("Save review to this session")
    if submitted:
        try:
            annotation = completed_review(
                review_template(snapshot, case), reviewer=reviewer, before_labels=before, after_labels=after,
                evidence_doc_ids=evidence, rationale=rationale, allowed_doc_ids=set(allowed_ids),
            )
            st.session_state.setdefault("human_reviews", {})[case["case_id"]] = annotation
            st.success("Review saved in this session. Download it below before closing the browser.")
        except ValueError as exc:
            st.error(str(exc))
    reviews = st.session_state.get("human_reviews", {})
    st.caption(f"{len(reviews)} explicit reviewer annotations in this session. Nothing is written to published results.")
    if reviews:
        st.download_button("Download session reviews", "".join(json.dumps(reviews[k]) + "\n" for k in sorted(reviews)),
                           "human_reviews.jsonl", mime="application/x-ndjson")


def render() -> None:
    st.caption("SAVED EXPERIMENTS / OFFLINE REPLAY")
    st.title("When does repair actually help?")
    st.write("Inspect the answer, the decision, and the evidence. Every outcome below was recorded in the "
             "held-out study—nothing is being generated live.")
    try:
        paths = [RELIABILITY_STUDY_DIR / name for name in (
            "manifest.json", "model.json", "test_model_lock.json", "development_outcomes.jsonl", "test_outcomes.jsonl",
        )] + [CORPUS_PATH, EXAMPLES_PATH]
        snapshot = saved_snapshot(tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"Cannot verify saved study artifacts: {exc}")
        st.info("Restore the published study files, corpus, and examples from the repository. No API call was attempted.")
        return

    cases = snapshot.cases
    st.caption("Frozen policy · 60 held-out questions / 300 states · HotpotQA · one provider seed")
    a, b, c, d = st.columns(4)
    a.metric("Saved states", len(cases))
    b.metric("EM recoveries", sum(c["transition"] == "Recovered" for c in cases))
    c.metric("EM regressions", sum(c["transition"] == "Harmed" for c in cases))
    d.metric("New API calls", 0)
    st.caption("Counts cover the full test set, not the current filter. Exact match is a lexical metric, not a factuality verdict.")
    with st.sidebar:
        st.divider()
        st.subheader("Explore cases")
        family = st.selectbox("Failure family", ["All", *FAMILY_LABELS], format_func=lambda x: FAMILY_LABELS.get(x, x))
        view = st.selectbox("Show", ["All cases", "Recovered", "Harmed", "Kept correct", "Missed recovery", "Format candidates"])
        search = st.text_input("Search question or ID")
        queue_only = st.checkbox("20-case review queue only")
        st.caption("Start with Recovered, then Harmed, then Kept correct. These views show both benefits and limits.")
    filtered = filter_cases(cases, family=family, view=view, search=search)
    if queue_only:
        queue_ids = {c["case_id"] for c in review_sample(cases)}
        filtered = [c for c in filtered if c["case_id"] in queue_ids]
    st.divider()
    if not filtered:
        st.info("No cases match these filters. Clear the search or choose All cases.")
        return
    by_id = {c["case_id"]: c for c in filtered}
    ids = list(by_id)
    linked = st.query_params.get("case")
    selected = st.selectbox(f"Case · {len(ids)} matching", ids,
                            index=ids.index(linked) if linked in ids else 0,
                            format_func=lambda key: f"{FAMILY_LABELS[by_id[key]['family']]} · {by_id[key]['question']}", key="replay_case")
    st.query_params["case"] = selected
    case, row = by_id[selected], snapshot.rows[selected]
    st.subheader(row["question"])
    st.caption(f"{selected} · {FAMILY_LABELS[case['family']]} · Frozen-policy outcome: {case['transition']}")
    st.markdown("**Reference answer**")
    st.write(row["gold"])
    decision = case["action"]
    utilities = case["predicted_utilities"]
    advantage = max(utilities.values()) - utilities["accept"]
    st.info(f"Frozen router → {ACTION_LABELS[decision]}. Best predicted gain over keeping the original: "
            f"{advantage:.3f}; required gain: {snapshot.policy.min_gain:.2f}. "
            "These are cost-adjusted utility scores, not calibrated confidence probabilities.")

    compare = st.selectbox("Compare original with saved action", list(ACTIONS), index=list(ACTIONS).index(decision),
                           format_func=ACTION_LABELS.get, key=f"compare_{selected}")
    outcome = row["outcomes"][compare]
    if compare != decision:
        st.caption("Counterfactual view: this is a saved alternative, not the frozen router's selected action.")
    usage = outcome["telemetry"]
    a, b, c, d = st.columns(4)
    a.metric("Recorded added calls", usage["llm_calls"])
    b.metric("Recorded tokens", f"{usage['total_tokens']:,}")
    c.metric("Estimated added cost", f"${usage['estimated_cost_usd']:.6f}")
    d.metric("Recorded wall time", f"{usage['wall_clock_seconds']:.2f}s")
    st.caption(f"Incremental action only; original generation excluded. Local throttle wait: "
               f"{usage.get('throttle_seconds', 0):.2f}s, included in wall time. Replay itself is free.")
    left, right = st.columns(2)
    with left:
        answer_card("Before · original output", row["initial"]["answer"], row["initial"]["query"],
                    case["before_metrics"], row["initial"]["docs"], case["support_doc_ids"])
    with right:
        answer_card(f"After · {ACTION_LABELS[compare]}", outcome["answer"], outcome["query"],
                    outcome["metrics"], evidence_for(row, compare, snapshot.corpus), case["support_doc_ids"])

    alternatives, audit, provenance = st.tabs(["All saved actions", "Error audit & review", "Provenance & export"])
    with alternatives:
        st.caption("Every action was observed during collection. The router saw only the initial state—not these outcomes.")
        st.dataframe([{
            "Action": ACTION_LABELS[action], "Selected": action == decision,
            "Predicted utility": round(utilities[action], 4),
            "EM": row["outcomes"][action]["metrics"]["exact_match"],
            "F1": round(row["outcomes"][action]["metrics"]["token_f1"], 3),
            "Calls": row["outcomes"][action]["telemetry"]["llm_calls"],
            "Tokens": row["outcomes"][action]["telemetry"]["total_tokens"],
            "USD": f"{row['outcomes'][action]['telemetry']['estimated_cost_usd']:.6f}",
            "Answer": row["outcomes"][action]["answer"],
        } for action in ACTIONS], hide_index=True, width="stretch")
        with st.expander("Gold-free input features"):
            st.json(case["features"])
    with audit:
        review_panel(snapshot, case)
    with provenance:
        st.write("Model, features, metrics, and evidence hashes were checked when loading the snapshot. "
                 "Published outcomes and policy parameters are read-only.")
        st.json(snapshot.provenance)
        st.code(f"?case={selected}", language=None)
        st.download_button("Download this case + evidence", case_export(snapshot, case),
                           f"case_{row['question_id']}_{row['family']}.json", mime="application/json")
