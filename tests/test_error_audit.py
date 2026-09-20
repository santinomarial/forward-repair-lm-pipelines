import json
import sys

import pytest

import error_audit as audit
from case_explorer import load_snapshot


@pytest.fixture(scope="module")
def snapshot():
    return load_snapshot()


def test_audit_reproducible_damage_first_and_not_human_reviewed(snapshot):
    summary, queue = audit.build_audit(snapshot)
    assert len(queue) == 20
    assert all(r["transition"] == "Harmed" for r in queue[:4])
    assert all(r["status"] == "pending" and not r["before_labels"] for r in queue)
    assert summary["groups"]["all"]["recovery_with_gold_already_present"] == 21
    assert summary["groups"]["all"]["missed_recovery"] == 15
    assert audit.review_sample(list(reversed(snapshot.cases))) == audit.review_sample(snapshot.cases)
    assert len(audit.review_sample(snapshot.cases, 1000)) == 300
    assert audit.review_sample([]) == []
    with pytest.raises(ValueError):
        audit.review_sample(snapshot.cases, 0)
    assert "0 are human-reviewed" in audit.audit_markdown(summary)


def test_review_requires_explicit_labels_rationale_and_evidence(snapshot):
    template = audit.review_template(snapshot, snapshot.cases[0])
    kwargs = dict(reviewer="Reviewer", before_labels=["format_only"], after_labels=["uncertain"],
                  evidence_doc_ids=["doc"], rationale="Supporting passage says …", allowed_doc_ids={"doc"})
    result = audit.completed_review(template, **kwargs)
    assert result["status"] == "reviewed"
    assert template["status"] == "pending"
    for change in ({"reviewer": ""}, {"rationale": ""}, {"before_labels": []},
                   {"after_labels": ["invented"]}, {"evidence_doc_ids": ["wrong"]},
                   {"after_labels": ["no_error"], "evidence_doc_ids": []}):
        with pytest.raises(ValueError):
            audit.completed_review(template, **(kwargs | change))


def test_audit_cli_preserves_separate_human_reviews(tmp_path, monkeypatch, snapshot):
    path = tmp_path / "human_reviews.jsonl"
    path.write_text("reviewer-owned-data\n")
    monkeypatch.setattr(audit, "load_snapshot", lambda _: snapshot)
    monkeypatch.setattr(sys, "argv", ["audit", "--output", str(tmp_path), "--sample-size", "8"])
    audit.main()
    assert path.read_text() == "reviewer-owned-data\n"
    assert len((tmp_path / "review_queue.jsonl").read_text().splitlines()) == 8
    assert json.loads((tmp_path / "summary.json").read_text())["review_sample"]["reviewed"] == 0
