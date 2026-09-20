import copy

import pytest

from outcome_routing import ACTIONS, OUTCOME_FEATURES, OutcomeRouter, fit_outcome_router, runtime_features, select_router


def row(i, split="train", winner="regenerate"):
    return {
        "question_id": str(i), "question": "Where is Paris?", "split": split,
        "family": "vague_query", "initial": {"query": "Paris", "answer": "unknown",
            "docs": [{"id": "x", "title": "Paris", "text": "Paris is in France.", "score": 2.0}]},
        "outcomes": {a: {"metrics": {"exact_match": int(a == winner)},
                         "telemetry": {"estimated_cost_usd": 0 if a == "accept" else .001}}
                     for a in ACTIONS},
    }


def test_router_learns_outcomes_not_stage_labels_and_serializes(tmp_path):
    rows = [row(i) for i in range(6)]
    model = fit_outcome_router(rows, ridge=1, cost_penalty=100)
    initial = rows[0]["initial"]
    assert model.decide(question="Where is Paris?", initial=initial) == "regenerate"
    assert len(runtime_features("Where is Paris?", initial)) == len(OUTCOME_FEATURES)
    changed = copy.deepcopy(initial)
    changed.update(gold="leaked", family="ignore_context", outcomes={"fake": "best"})
    assert model.decide(question="Where is Paris?", initial=changed) == "regenerate"
    path = tmp_path / "model.json"
    model.save(path, {"test": True})
    assert OutcomeRouter.load(path) == model
    with pytest.raises(FileExistsError):
        model.save(path, {})


def test_cost_penalty_changes_choice_when_success_is_equal():
    rows = [row(i) for i in range(5)]
    for r in rows:
        for action in ACTIONS:
            r["outcomes"][action]["metrics"]["exact_match"] = 1
    model = fit_outcome_router(rows, ridge=1, cost_penalty=100)
    assert model.decide(question=rows[0]["question"], initial=rows[0]["initial"]) == "accept"


def test_fit_and_selection_reject_split_or_family_leakage():
    with pytest.raises(ValueError, match="training questions"):
        fit_outcome_router([row(1, "test")], ridge=1, cost_penalty=100)
    unseen = row(2)
    unseen["family"] = "retrieval_rank_dropout"
    with pytest.raises(ValueError, match="seen families"):
        fit_outcome_router([unseen], ridge=1, cost_penalty=100)
    with pytest.raises(ValueError, match="overlap"):
        select_router([row(1)], [row(1, "validation")])
    with pytest.raises(ValueError, match="validation"):
        select_router([row(1)], [row(2, "test")])


def test_validation_grid_has_no_test_outcomes():
    model, selection = select_router([row(i) for i in range(8)], [row(9, "validation")])
    assert len(selection["candidates"]) == 12
    assert model.decide(question="Where is Paris?", initial=row(0)["initial"]) == "regenerate"
    with pytest.raises(ValueError, match="features"):
        model.decide_features([float("nan")] * len(OUTCOME_FEATURES))


@pytest.mark.parametrize("ridge,penalty", [(0, 1), (1, -1)])
def test_invalid_fit_configuration(ridge, penalty):
    with pytest.raises(ValueError):
        fit_outcome_router([row(1)], ridge=ridge, cost_penalty=penalty)
