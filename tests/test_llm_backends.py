from unittest.mock import Mock, patch

import pytest

from llm_backends import OllamaBackend, OpenAIBackend, build_llm_backend


@patch("llm_backends.dspy.LM")
def test_openai_backend_builds_deterministic_and_seeded_models(mock_lm):
    deterministic = Mock(name="deterministic")
    stochastic = Mock(name="stochastic")
    mock_lm.side_effect = [deterministic, stochastic]
    backend = OpenAIBackend(model="gpt-test", api_key="test-key")

    models = backend.create_models(seed=17)

    assert models.deterministic is deterministic
    assert models.stochastic is stochastic
    assert mock_lm.call_args_list[0].kwargs == {
        "model": "openai/gpt-test",
        "api_key": "test-key",
        "temperature": 0,
        "max_tokens": 300,
    }
    assert mock_lm.call_args_list[1].kwargs["seed"] == 17


@patch("llm_backends.dspy.LM")
def test_ollama_backend_uses_local_chat_endpoint(mock_lm):
    backend = OllamaBackend("llama3.2:3b", "http://localhost:11434/")

    backend.create_lm(temperature=0)

    assert mock_lm.call_args.kwargs == {
        "model": "ollama_chat/llama3.2:3b",
        "api_base": "http://localhost:11434",
        "temperature": 0,
        "max_tokens": 300,
    }


def test_openai_backend_requires_api_key():
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        OpenAIBackend(model="gpt-test", api_key=None)


def test_backend_factory_selects_provider_and_rejects_unknown():
    assert isinstance(
        build_llm_backend("ollama", "llama3.2:3b"),
        OllamaBackend,
    )
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        build_llm_backend("other", "model")
