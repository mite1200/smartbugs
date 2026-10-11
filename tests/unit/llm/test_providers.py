import json
from types import SimpleNamespace

import pytest
import requests

from sb.llm.models import ProviderConfig
from sb.llm.providers import (
    GeminiProvider,
    OpenAIChatProvider,
    OpenAIResponsesProvider,
    ProviderError,
    TransientProviderError,
    create_provider,
)


class FakeSession:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.error:
            raise self.error
        return self.response


def fake_response(data, status=200, headers=None):
    return SimpleNamespace(status_code=status, text=json.dumps(data), headers=headers or {})


def config(provider="openai", **overrides):
    value = {
        "provider": provider,
        "model": "model-id",
        "api_key_env": "TEST_LLM_KEY",
        "base_url": "https://provider.example/v1",
        "max_output_tokens": 500,
    }
    value.update(overrides)
    return ProviderConfig.from_dict(value)


def test_missing_api_key_has_useful_error(monkeypatch):
    monkeypatch.delenv("TEST_LLM_KEY", raising=False)
    with pytest.raises(ProviderError, match="TEST_LLM_KEY"):
        create_provider(config())


def test_openai_uses_responses_api(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "id": "resp_1",
                "model": "model-version",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
                "usage": {
                    "input_tokens": 10,
                    "output_tokens": 4,
                    "output_tokens_details": {"reasoning_tokens": 2},
                },
            }
        )
    )
    provider = create_provider(config(), session)
    assert isinstance(provider, OpenAIResponsesProvider)
    result = provider.infer("prompt")
    url, request = session.calls[0]
    assert url.endswith("/responses")
    assert request["json"]["input"] == "prompt"
    assert "messages" not in request["json"]
    assert "temperature" not in request["json"]
    assert request["json"]["store"] is False
    assert result.request_id == "resp_1"
    assert result.reasoning_tokens == 2


def test_openai_responses_store_false_cannot_be_overridden(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "id": "resp_1",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
                "usage": {},
            }
        )
    )
    provider = create_provider(config(options={"request_body": {"store": True}}), session)
    provider.infer("prompt")
    assert session.calls[0][1]["json"]["store"] is False


def test_provider_redacts_api_key_if_response_echoes_it(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret-value")
    session = FakeSession(
        fake_response(
            {
                "id": "resp_1",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": "secret-value must not persist"}
                        ],
                    }
                ],
            }
        )
    )
    result = create_provider(config(), session).infer("prompt")
    assert "secret-value" not in result.raw_response
    assert "secret-value" not in result.text
    assert "[REDACTED]" in result.raw_response


def test_openai_compatible_defaults_to_chat_completions(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "id": "chat_1",
                "model": "qwen-version",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            }
        )
    )
    provider = create_provider(config("openai_compatible"), session)
    assert isinstance(provider, OpenAIChatProvider)
    result = provider.infer("prompt")
    assert session.calls[0][0].endswith("/chat/completions")
    assert "temperature" not in session.calls[0][1]["json"]
    assert "reasoning_effort" not in session.calls[0][1]["json"]
    assert result.output_tokens == 3


def test_openai_compatible_includes_normalized_reasoning_effort(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }
        )
    )
    provider = create_provider(config("openai_compatible", reasoning={"effort": "medium"}), session)
    provider.infer("prompt")
    assert session.calls[0][1]["json"]["reasoning_effort"] == "medium"


def test_openai_compatible_normalized_reasoning_effort_overrides_request_body(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }
        )
    )
    provider = create_provider(
        config(
            "openai_compatible",
            reasoning={"effort": "medium"},
            options={"request_body": {"reasoning_effort": "low"}},
        ),
        session,
    )
    provider.infer("prompt")
    assert session.calls[0][1]["json"]["reasoning_effort"] == "medium"


def test_openai_compatible_explicit_temperature_remains_supported(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "choices": [{"message": {"content": "{}"}}],
                "usage": {},
            }
        )
    )
    provider = create_provider(config("openai_compatible", temperature=0.25), session)
    provider.infer("prompt")
    assert session.calls[0][1]["json"]["temperature"] == 0.25


def test_gemini_request_and_metadata(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "responseId": "gem_1",
                "modelVersion": "gemini-version",
                "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 2,
                    "thoughtsTokenCount": 1,
                },
            }
        )
    )
    provider = create_provider(
        config("gemini", model="gemini-3.8-flash", reasoning={"effort": "medium"}), session
    )
    assert isinstance(provider, GeminiProvider)
    result = provider.infer("prompt")
    assert ":generateContent" in session.calls[0][0]
    request = session.calls[0][1]["json"]
    generation_config = request["generationConfig"]
    assert generation_config["thinkingConfig"] == {"thinkingLevel": "medium"}
    assert "temperature" not in generation_config
    assert "topP" not in generation_config
    assert "topK" not in generation_config
    assert result.reasoning_tokens == 1
    assert result.metadata == {"finish_reason": "STOP"}


def test_other_gemini_models_keep_explicit_temperature_support(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "candidates": [{"content": {"parts": [{"text": "{}"}]}}],
                "usageMetadata": {},
            }
        )
    )
    provider = create_provider(config("gemini", model="other-gemini", temperature=0.4), session)
    provider.infer("prompt")
    assert session.calls[0][1]["json"]["generationConfig"]["temperature"] == 0.4


@pytest.mark.parametrize("effort", ["minimal", "none", "extreme", 1])
def test_gemini_38_rejects_unsupported_thinking_levels(effort):
    with pytest.raises(
        ValueError,
        match="gemini-3.8-flash reasoning.effort must be one of: low, medium, high",
    ):
        config("gemini", model="gemini-3.8-flash", reasoning={"effort": effort})


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0},
        {"options": {"generation_config": {"top_p": 0.9}}},
        {"options": {"generation_config": {"topK": 20}}},
        {"options": {"request_body": {"temperature": 0.2}}},
        {"options": {"request_body": {"top_p": 0.9}}},
        {"options": {"generation_config": {"thinkingConfig": {"thinkingLevel": "low"}}}},
    ],
)
def test_gemini_38_rejects_sampling_parameters(overrides):
    with pytest.raises(ValueError, match="does not support sampling configuration"):
        config("gemini", model="gemini-3.8-flash", **overrides)


def test_native_openai_reasoning_effort_maps_without_sampling(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    session = FakeSession(
        fake_response(
            {
                "id": "resp_1",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "{}"}]}],
                "usage": {},
            }
        )
    )
    provider = create_provider(
        config("openai", model="gpt-6-luna", reasoning={"effort": "medium"}), session
    )
    provider.infer("prompt")
    payload = session.calls[0][1]["json"]
    assert payload["model"] == "gpt-6-luna"
    assert payload["reasoning"] == {"effort": "medium"}
    assert "temperature" not in payload
    assert "top_p" not in payload


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0},
        {"options": {"request_body": {"top_p": 0.9}}},
        {"options": {"request_body": {"temperature": 0.2}}},
    ],
)
def test_native_openai_reasoning_rejects_sampling_parameters(overrides):
    with pytest.raises(ValueError, match="cannot be combined"):
        config("openai", reasoning={"effort": "medium"}, **overrides)


def test_native_openai_rejects_reasoning_override_in_request_body():
    with pytest.raises(ValueError, match="top-level reasoning mapping"):
        config(
            "openai",
            reasoning={"effort": "medium"},
            options={"request_body": {"reasoning": {"effort": "low"}}},
        )


def test_transient_http_and_connection_failures_are_distinguished(monkeypatch):
    monkeypatch.setenv("TEST_LLM_KEY", "secret")
    provider = create_provider(config(), FakeSession(fake_response({}, status=429)))
    with pytest.raises(TransientProviderError):
        provider.infer("prompt")
    provider = create_provider(config(), FakeSession(error=requests.ConnectionError("reset")))
    with pytest.raises(TransientProviderError):
        provider.infer("prompt")
