from abc import ABC, abstractmethod
from dataclasses import dataclass

import dspy


@dataclass(frozen=True)
class LMSet:
    deterministic: dspy.LM
    stochastic: dspy.LM


class LLMBackend(ABC):
    """Factory interface for the deterministic and corruption LMs."""

    def __init__(self, model: str, max_tokens: int = 300):
        self.model = model
        self.max_tokens = max_tokens

    @abstractmethod
    def create_lm(self, temperature: float, seed: int | None = None) -> dspy.LM:
        """Create one DSPy LM configured for this provider."""

    def create_models(self, seed: int) -> LMSet:
        return LMSet(
            deterministic=self.create_lm(temperature=0),
            stochastic=self.create_lm(temperature=0.7, seed=seed),
        )

    def configure(self, seed: int) -> dspy.LM:
        models = self.create_models(seed)
        dspy.configure(lm=models.deterministic)
        return models.stochastic


class OpenAIBackend(LLMBackend):
    def __init__(self, model: str, api_key: str | None, max_tokens: int = 300):
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
        super().__init__(model=model, max_tokens=max_tokens)
        self.api_key = api_key

    def create_lm(self, temperature: float, seed: int | None = None) -> dspy.LM:
        model = self.model if self.model.startswith("openai/") else f"openai/{self.model}"
        kwargs = {
            "model": model,
            "api_key": self.api_key,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        return dspy.LM(**kwargs)


class OllamaBackend(LLMBackend):
    def __init__(self, model: str, api_base: str, max_tokens: int = 300):
        super().__init__(model=model, max_tokens=max_tokens)
        self.api_base = api_base.rstrip("/")

    def create_lm(self, temperature: float, seed: int | None = None) -> dspy.LM:
        model = (
            self.model
            if self.model.startswith(("ollama/", "ollama_chat/"))
            else f"ollama_chat/{self.model}"
        )
        kwargs = {
            "model": model,
            "api_base": self.api_base,
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            kwargs["seed"] = seed
        return dspy.LM(**kwargs)


def build_llm_backend(
    backend: str,
    model: str,
    *,
    openai_api_key: str | None = None,
    ollama_api_base: str = "http://localhost:11434",
    max_tokens: int = 300,
) -> LLMBackend:
    if backend == "openai":
        return OpenAIBackend(model=model, api_key=openai_api_key, max_tokens=max_tokens)
    if backend == "ollama":
        return OllamaBackend(model=model, api_base=ollama_api_base, max_tokens=max_tokens)
    raise ValueError(f"Unknown LLM backend: {backend!r}")
