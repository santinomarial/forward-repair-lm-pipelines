from types import SimpleNamespace

from experiment import summarize_instrumentation
from telemetry import LMUsageSnapshot


def test_lm_usage_snapshot_counts_new_history_entries_only():
    first = SimpleNamespace(
        history=[
            {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}, "cost": 0.01}
        ]
    )
    second = SimpleNamespace(history=[])
    snapshot = LMUsageSnapshot([first, second])
    first.history.append(
        {
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            "cost": 0.02,
        }
    )
    second.history.append(
        {
            "usage": {"input_tokens": 5, "output_tokens": 1},
            "cost": None,
        }
    )

    assert snapshot.finish() == {
        "llm_calls": 2,
        "prompt_tokens": 25,
        "completion_tokens": 5,
        "total_tokens": 30,
        "estimated_cost_usd": 0.02,
        "priced_calls": 1,
        "estimated_token_calls": 0,
    }


def test_lm_usage_snapshot_estimates_tokens_when_cached_usage_is_missing():
    model = SimpleNamespace(history=[])
    snapshot = LMUsageSnapshot([model])
    model.history.append(
        {
            "model": "openai/gpt-4o-mini",
            "messages": [{"role": "user", "content": "Reply with OK"}],
            "outputs": ["OK"],
            "usage": {},
            "cost": None,
        }
    )

    result = snapshot.finish()

    assert result["prompt_tokens"] > 0
    assert result["completion_tokens"] > 0
    assert result["total_tokens"] == result["prompt_tokens"] + result["completion_tokens"]
    assert result["estimated_token_calls"] == 1


def _row(wall: float, calls: int, tokens: int, cost: float) -> dict:
    telemetry = {
        "wall_clock_seconds": wall,
        "latency_seconds": {
            "query_generation": wall * 0.2,
            "retrieval": wall * 0.1,
            "answer_generation": wall * 0.7,
        },
        "llm_calls": calls,
        "prompt_tokens": tokens - 2,
        "completion_tokens": 2,
        "total_tokens": tokens,
        "estimated_cost_usd": cost,
    }
    return {"baseline": {"telemetry": telemetry}}


def test_summarize_instrumentation_aggregates_totals_and_per_example():
    result = summarize_instrumentation(
        [_row(1.0, 2, 100, 0.01), _row(3.0, 2, 200, 0.03)],
        ["baseline"],
    )["baseline"]

    assert result["llm_calls"] == {"total": 4, "per_example": 2.0}
    assert result["tokens"]["total"] == 300
    assert result["tokens"]["per_example"] == 150
    assert result["estimated_cost_usd"]["total"] == 0.04
    assert result["wall_clock_seconds"] == {"total": 4.0, "per_example": 2.0}
    assert result["stage_latency_seconds"]["retrieval"]["per_example"] == 0.2
