from stratified_analysis import classify, compute_seed_stats


def test_classify_known_example_structures():
    corpus = {
        "one": "The capital is Paris.",
        "two": "France is in Europe.",
        "three": "Paris is also a person's name.",
    }

    assert classify("Paris", ["one", "two"], corpus) == "single-hop-sufficient"
    assert classify("Paris", ["one", "three"], corpus) == "genuinely-multi-hop"
    assert classify("London", ["one", "two"], corpus) == "answer-not-in-support"
    assert classify(" YES ", ["one", "two"], corpus) == "genuinely-multi-hop"


def _condition(exact_match: int) -> dict:
    return {
        "metrics": {
            "exact_match": exact_match,
            "contains_answer": exact_match,
            "recall_at_k": 1,
            "all_support_recall_at_k": 1,
        }
    }


def test_compute_seed_stats_assigns_rows_to_expected_strata():
    corpus = {
        "single": "The answer is Alpha.",
        "other": "No answer here.",
        "multi-a": "Beta appears here.",
        "multi-b": "This also says Beta.",
    }
    rows = [
        {
            "id": "s",
            "gold": "Alpha",
            "support_doc_ids": ["single", "other"],
            "baseline": _condition(1),
            "corrupted": _condition(0),
            "repaired": _condition(1),
        },
        {
            "id": "m",
            "gold": "Beta",
            "support_doc_ids": ["multi-a", "multi-b"],
            "baseline": _condition(1),
            "corrupted": _condition(0),
            "repaired": _condition(0),
        },
        {
            "id": "y",
            "gold": "yes",
            "support_doc_ids": ["single", "other"],
            "baseline": _condition(1),
            "corrupted": _condition(0),
            "repaired": _condition(1),
        },
        {
            "id": "x",
            "gold": "Missing",
            "support_doc_ids": ["single", "other"],
            "baseline": _condition(0),
            "corrupted": _condition(0),
            "repaired": _condition(0),
        },
    ]

    stats = compute_seed_stats(rows, corpus, ["baseline", "corrupted", "repaired"])

    assert stats["n"] == {
        "total": 4,
        "excluded": 1,
        "single_hop": 1,
        "multi_hop": 2,
        "yesno": 1,
        "bridge": 1,
    }
    assert stats["recovery"]["single_hop"]["repaired"]["rate"] == 1.0
    assert stats["recovery"]["multi_hop"]["repaired"]["rate"] == 0.5
