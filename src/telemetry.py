"""LLM usage accounting based on DSPy's per-model request history."""

from collections.abc import Iterable
from typing import Any

from litellm import token_counter


def summarize_instrumentation(rows: list[dict], modes: list[str]) -> dict:
    result: dict = {
        "cost_note": (
            "estimated_cost_usd is DSPy/LiteLLM provider-reported cost; "
            "cache hits and local calls may report zero"
        )
    }
    for mode in modes:
        telemetry = [row[mode]["telemetry"] for row in rows]
        n = len(telemetry)
        latency_keys = ["query_generation", "retrieval", "answer_generation"]
        total_calls = sum(int(item["llm_calls"]) for item in telemetry)
        total_tokens = sum(int(item["total_tokens"]) for item in telemetry)
        total_cost = sum(float(item["estimated_cost_usd"]) for item in telemetry)
        total_wall = sum(float(item["wall_clock_seconds"]) for item in telemetry)
        result[mode] = {
            "examples": n,
            "llm_calls": {"total": total_calls, "per_example": total_calls / n},
            "tokens": {
                "prompt_total": sum(int(item["prompt_tokens"]) for item in telemetry),
                "completion_total": sum(
                    int(item["completion_tokens"]) for item in telemetry
                ),
                "total": total_tokens,
                "per_example": total_tokens / n,
            },
            "estimated_cost_usd": {
                "total": total_cost,
                "per_example": total_cost / n,
            },
            "wall_clock_seconds": {
                "total": total_wall,
                "per_example": total_wall / n,
            },
            "stage_latency_seconds": {
                key: {
                    "total": sum(
                        float(item["latency_seconds"][key]) for item in telemetry
                    ),
                    "per_example": sum(
                        float(item["latency_seconds"][key]) for item in telemetry
                    )
                    / n,
                }
                for key in latency_keys
            },
        }
    return result


def _estimate_tokens(entry: dict) -> tuple[int, int]:
    model = entry.get("model", "")
    messages = entry.get("messages")
    prompt = entry.get("prompt")
    outputs = entry.get("outputs") or []
    prompt_tokens = token_counter(model=model, messages=messages, text=prompt)
    output_text = "\n".join(
        output if isinstance(output, str) else str(output) for output in outputs
    )
    completion_tokens = token_counter(model=model, text=output_text) if output_text else 0
    return prompt_tokens, completion_tokens


class LMUsageSnapshot:
    def __init__(self, models: Iterable[Any]):
        self.models = list(models)
        self.starts = {id(model): len(model.history) for model in self.models}

    def finish(self) -> dict[str, int | float]:
        entries = []
        for model in self.models:
            entries.extend(model.history[self.starts[id(model)] :])

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        estimated_cost = 0.0
        priced_calls = 0
        estimated_token_calls = 0

        for entry in entries:
            usage = entry.get("usage") or {}
            prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
            completion = int(
                usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
            )
            total = usage.get("total_tokens")
            if not total and prompt == 0 and completion == 0 and (
                entry.get("messages") or entry.get("prompt")
            ):
                prompt, completion = _estimate_tokens(entry)
                estimated_token_calls += 1
            prompt_tokens += prompt
            completion_tokens += completion
            total_tokens += int(total) if total else prompt + completion
            if entry.get("cost") is not None:
                estimated_cost += float(entry["cost"])
                priced_calls += 1

        return {
            "llm_calls": len(entries),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost,
            "priced_calls": priced_calls,
            "estimated_token_calls": estimated_token_calls,
        }
