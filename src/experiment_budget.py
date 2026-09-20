"""Persistent, thread-safe conservative request reservations for GPT-4o-mini.

Reservations are never refunded, including failed requests. Automatic transport
retries are disabled. Every DSPy adapter fallback must reserve again. This is a
local experiment guard, not an account-wide billing limit.
"""

import json
import math
import os
from pathlib import Path
import threading
import time
from collections import deque

import dspy
from litellm import token_counter


class BudgetExceeded(RuntimeError):
    pass


class TokenRateLimiter:
    """Rolling-minute token reservations shared across worker threads."""

    def __init__(self, limit: int = 180_000, recent: list[dict] | None = None):
        self.limit = limit
        self.lock = threading.Lock()
        self.events = deque(
            (float(entry["timestamp"]), int(entry["rate_tokens"]))
            for entry in (recent or [])
            if entry.get("rate_tokens", 0) > 0 and time.time() - entry.get("timestamp", 0) < 60
        )

    def acquire(self, tokens: int) -> float:
        if not 0 <= tokens <= self.limit:
            raise ValueError("request exceeds local token rate limit")
        started = time.monotonic()
        while True:
            with self.lock:
                now = time.time()
                while self.events and self.events[0][0] <= now - 60:
                    self.events.popleft()
                if sum(count for _, count in self.events) + tokens <= self.limit:
                    self.events.append((now, tokens))
                    return time.monotonic() - started
                delay = max(.01, self.events[0][0] + 60 - now)
            time.sleep(min(delay, 1.0))


class ExperimentBudget:
    def __init__(self, path: Path, limit_usd: float):
        if not math.isfinite(limit_usd) or limit_usd <= 0:
            raise ValueError("budget must be finite and positive")
        self.path = path
        self.limit_usd = limit_usd
        self.lock = threading.Lock()
        self.reserved_usd = 0.0
        self.requests = 0
        entries = []
        if path.exists():
            for line in path.read_text().splitlines():
                entry = json.loads(line)
                entries.append(entry)
                if entry["limit_usd"] != limit_usd:
                    raise ValueError("cannot change an existing ledger's budget")
                amount = float(entry["reserved_usd"])
                if not math.isfinite(amount) or amount <= 0:
                    raise ValueError("invalid reservation ledger")
                self.reserved_usd += amount
                self.requests += 1
        if self.reserved_usd > limit_usd:
            raise ValueError("ledger already exceeds configured budget")
        self.rate_limiter = TokenRateLimiter(recent=entries)

    def reserve(self, amount: float, *, case_id: str, rate_tokens: int = 0) -> float:
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("reservation must be finite and positive")
        waited = self.rate_limiter.acquire(rate_tokens) if rate_tokens else 0.0
        with self.lock:
            if self.reserved_usd + amount > self.limit_usd:
                raise BudgetExceeded(f"Stopping before request: ${self.limit_usd:.2f} reservation cap reached")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "case_id": case_id, "request": self.requests + 1,
                    "reserved_usd": amount, "limit_usd": self.limit_usd,
                    "rate_tokens": rate_tokens, "timestamp": time.time(),
                }) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self.reserved_usd += amount
            self.requests += 1
        return waited


class BudgetedLM(dspy.LM):
    def __init__(self, budget: ExperimentBudget, case_id: str, *, api_key: str, seed: int):
        super().__init__(
            "openai/gpt-4o-mini", api_key=api_key, temperature=0, max_tokens=300,
            seed=seed, cache=False, num_retries=0, timeout=60,
        )
        self.experiment_budget = budget
        self.case_id = case_id
        self.throttle_seconds = 0.0

    def forward(self, prompt=None, messages=None, **kwargs):
        settings = {**self.kwargs, **kwargs}
        if settings.get("n", 1) != 1:
            raise ValueError("budgeted study permits only one completion per request")
        max_tokens = int(settings.get("max_tokens", 300))
        if not 0 < max_tokens <= 300:
            raise ValueError("study output limit must be 1–300 tokens")
        # Count the actual composed request, including any response schema.
        extra = json.dumps(settings.get("response_format", {}), default=str)
        prompt_tokens = token_counter(model=self.model, messages=messages, text=prompt)
        extra_tokens = token_counter(model=self.model, text=extra)
        # Uncached list pricing, full output allowance, plus padding and 25% headroom.
        reservation = 1.25 * ((prompt_tokens + extra_tokens + 256) * 0.15 + max_tokens * 0.60) / 1_000_000
        self.throttle_seconds += self.experiment_budget.reserve(
            reservation, case_id=self.case_id, rate_tokens=prompt_tokens + extra_tokens + max_tokens,
        )
        return super().forward(prompt=prompt, messages=messages, **kwargs)
