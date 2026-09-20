from concurrent.futures import ThreadPoolExecutor
import json

import dspy
import pytest

from experiment_budget import BudgetExceeded, BudgetedLM, ExperimentBudget, TokenRateLimiter


def test_budget_persists_and_refuses_overspend(tmp_path):
    path = tmp_path / "budget.jsonl"
    budget = ExperimentBudget(path, 1)
    budget.reserve(.6, case_id="one")
    with pytest.raises(BudgetExceeded):
        budget.reserve(.5, case_id="two")
    reloaded = ExperimentBudget(path, 1)
    assert reloaded.reserved_usd == .6
    assert reloaded.requests == 1
    with pytest.raises(ValueError, match="change"):
        ExperimentBudget(path, 2)


def test_concurrent_reservations_cannot_exceed_limit(tmp_path):
    budget = ExperimentBudget(tmp_path / "budget.jsonl", 1)

    def reserve(_):
        try:
            budget.reserve(.3, case_id="test")
            return True
        except BudgetExceeded:
            return False

    with ThreadPoolExecutor(max_workers=3) as pool:
        assert sum(pool.map(reserve, range(12))) == 3
    assert budget.reserved_usd == pytest.approx(.9)


def test_lm_reserves_before_request_including_failures(tmp_path, monkeypatch):
    budget = ExperimentBudget(tmp_path / "budget.jsonl", 1)
    lm = BudgetedLM(budget, "case", api_key="not-real", seed=0)
    assert lm.num_retries == 0
    assert lm.cache is False

    def fake_forward(*args, **kwargs):
        assert budget.requests == 1
        raise RuntimeError("provider failure")

    monkeypatch.setattr(dspy.LM, "forward", fake_forward)
    with pytest.raises(RuntimeError, match="provider failure"):
        lm.forward(messages=[{"role": "user", "content": "hello"}])
    assert budget.reserved_usd > 0
    with pytest.raises(ValueError, match="one completion"):
        lm.forward(prompt="hello", n=2)
    assert budget.requests == 1


def test_exhausted_budget_never_calls_provider(tmp_path, monkeypatch):
    budget = ExperimentBudget(tmp_path / "budget.jsonl", .000001)
    lm = BudgetedLM(budget, "case", api_key="not-real", seed=0)
    monkeypatch.setattr(dspy.LM, "forward", lambda *a, **k: pytest.fail("request must be blocked"))
    with pytest.raises(BudgetExceeded):
        lm.forward(prompt="hello")


def test_five_dollar_cap_is_shared_across_resumed_phases(tmp_path, monkeypatch):
    path = tmp_path / "natural_budget.jsonl"
    budget = ExperimentBudget(path, 5)
    budget.reserve(2, case_id="pilot")
    budget = ExperimentBudget(path, 5)
    budget.reserve(2.999999, case_id="development")
    lm = BudgetedLM(ExperimentBudget(path, 5), "heldout", api_key="not-real", seed=0)
    monkeypatch.setattr(dspy.LM, "forward", lambda *a, **k: pytest.fail("cap must block provider"))
    with pytest.raises(BudgetExceeded):
        lm.forward(prompt="hello")
    assert len(path.read_text().splitlines()) == 2
    assert ExperimentBudget(path, 5).reserved_usd < 5
    with pytest.raises(ValueError, match="change"):
        ExperimentBudget(path, 6)


@pytest.mark.parametrize("amount", [-1, 0, float("nan"), float("inf")])
def test_invalid_budget_amounts(tmp_path, amount):
    with pytest.raises(ValueError):
        ExperimentBudget(tmp_path / "budget.jsonl", amount)
    budget = ExperimentBudget(tmp_path / "good.jsonl", 1)
    with pytest.raises(ValueError):
        budget.reserve(amount, case_id="test")


def test_corrupt_ledger_is_not_silently_reset(tmp_path):
    path = tmp_path / "budget.jsonl"
    path.write_text(json.dumps({"limit_usd": 1, "reserved_usd": -1}))
    with pytest.raises(ValueError):
        ExperimentBudget(path, 1)


def test_rate_limiter_waits_and_restores_recent_history(monkeypatch):
    import experiment_budget as module
    clock = [100.0]
    monkeypatch.setattr(module.time, "time", lambda: clock[0])
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay))
    limiter = TokenRateLimiter(limit=10, recent=[{"timestamp": 90, "rate_tokens": 8}])
    assert limiter.acquire(3) == 50
    assert sum(value for _, value in limiter.events) == 3
    with pytest.raises(ValueError):
        limiter.acquire(11)
