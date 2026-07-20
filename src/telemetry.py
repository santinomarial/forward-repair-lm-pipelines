"""LLM usage accounting based on DSPy's per-model request history."""

from collections.abc import Iterable
from typing import Any

from litellm import token_counter


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
