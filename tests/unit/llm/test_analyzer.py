import json

import pytest

from sb.llm.analyzer import LLMAnalyzer
from sb.llm.models import ProviderConfig, ProviderResponse
from sb.llm.profiles import load_profile
from sb.llm.providers import GeminiProvider, OpenAIResponsesProvider, TransientProviderError


class FakeHTTPResponse:
    def __init__(self, data, status_code=200, headers=None):
        self.status_code = status_code
        self.text = json.dumps(data)
        self.headers = headers or {}


class FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.response


class FakeProvider:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.prompts = []

    def infer(self, prompt):
        self.prompts.append(prompt)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def provider_config(**overrides):
    value = {
        "provider": "openai",
        "model": "exact-model",
        "api_key_env": "UNUSED_IN_FAKE",
        "retry_count": 2,
        "retry_backoff_seconds": 1,
        "recommended_concurrency": 3,
    }
    value.update(overrides)
    return ProviderConfig.from_dict(value)


def valid_response(raw=None):
    text = raw or json.dumps(
        {
            "classification": "Safe",
            "vulnerable_functions": [],
            "reentered_functions": [],
            "reentrancy_type": None,
            "relevant_lines": [],
            "explanation": "No harmful re-entry exists.",
        }
    )
    return ProviderResponse(
        raw_response=json.dumps({"provider": "body"}),
        text=text,
        request_id="request-1",
        input_tokens=100,
        output_tokens=20,
        reasoning_tokens=5,
    )


def test_transient_failures_retry_with_identical_prompt_and_identity():
    provider = FakeProvider([TransientProviderError("429"), valid_response()])
    sleeps = []
    analyzer = LLMAnalyzer(
        load_profile("reentrancy"),
        provider_config(),
        provider=provider,
        sleep=sleeps.append,
        jitter=lambda: 0,
    )
    result = analyzer.analyze(
        "contract A {}\n",
        "A.sol",
        "zero_shot",
        {"repeat_index": 2, "authorization": "opaque-secret"},
        "fixed-id",
    )
    assert len(provider.prompts) == 2
    assert provider.prompts[0] == provider.prompts[1]
    assert sleeps == [1]
    provenance = result["provenance"]
    assert provenance["inference_id"] == "fixed-id"
    assert provenance["api_retry_count"] == 1
    assert provenance["metadata"] == {
        "repeat_index": 2,
        "authorization": "[REDACTED]",
    }
    assert provenance["usage"]["reasoning_tokens"] == 5


def test_bounded_retry_count_is_respected():
    provider = FakeProvider([TransientProviderError("a"), TransientProviderError("b")])
    analyzer = LLMAnalyzer(
        load_profile("reentrancy"),
        provider_config(retry_count=1),
        provider=provider,
        sleep=lambda _: None,
        jitter=lambda: 0,
    )
    result = analyzer.analyze("contract A {}", "A.sol", "zero_shot")
    assert len(provider.prompts) == 2
    assert result["status"] == "technical_error"
    assert result["provenance"]["technical_error"]["type"] == "TransientProviderError"


def test_malformed_model_output_is_not_retried():
    provider = FakeProvider([valid_response("not json")])
    analyzer = LLMAnalyzer(load_profile("reentrancy"), provider_config(), provider=provider)
    result = analyzer.analyze("contract A {}", "A.sol", "zero_shot")
    assert len(provider.prompts) == 1
    assert result["status"] == "invalid_output"
    assert result["provenance"]["raw_model_text"] == "not json"


@pytest.mark.parametrize(
    ("body", "expected_message", "expected_finish_reason"),
    [
        (
            {
                "candidates": [
                    {
                        "content": {"parts": [{"thought": True}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 20,
                    "candidatesTokenCount": 0,
                    "thoughtsTokenCount": 10,
                },
                "modelVersion": "gemini-3.8-flash",
                "responseId": "response-empty",
            },
            "Gemini response contains empty candidate text",
            "STOP",
        ),
        (
            {
                "candidates": [{"finishReason": "MAX_TOKENS"}],
                "usageMetadata": {
                    "promptTokenCount": 30,
                    "candidatesTokenCount": 100,
                    "thoughtsTokenCount": 90,
                },
                "modelVersion": "gemini-3.8-flash",
                "responseId": "response-max-tokens",
            },
            "Gemini response contains no candidate text",
            "MAX_TOKENS",
        ),
        (
            {
                "candidates": [],
                "promptFeedback": {
                    "blockReason": "SAFETY",
                    "safetyRatings": [{"category": "TEST", "probability": "HIGH"}],
                },
                "usageMetadata": {"promptTokenCount": 15},
                "modelVersion": "gemini-3.8-flash",
                "responseId": "response-blocked",
            },
            "Gemini response contains no candidate text",
            None,
        ),
    ],
)
def test_gemini_interpretation_errors_preserve_http_response_in_provenance(
    monkeypatch, body, expected_message, expected_finish_reason
):
    monkeypatch.setenv("TEST_GEMINI_KEY", "secret-key")
    response = FakeHTTPResponse(body, headers={"x-request-id": "header-request-id"})
    session = FakeSession(response)
    config = ProviderConfig.from_dict(
        {
            "provider": "gemini",
            "model": "gemini-3.8-flash",
            "api_key_env": "TEST_GEMINI_KEY",
            "reasoning": {"effort": "medium"},
            "retry_count": 2,
        }
    )
    analyzer = LLMAnalyzer(
        load_profile("reentrancy"), config, provider=GeminiProvider(config, session)
    )
    result = analyzer.analyze("contract A {}", "A.sol", "zero_shot")
    provenance = result["provenance"]
    assert result["status"] == "technical_error"
    assert provenance["technical_error"] == {
        "type": "ProviderError",
        "message": expected_message,
    }
    assert json.loads(provenance["raw_provider_response"]) == body
    assert provenance["raw_model_text"] is None
    assert provenance["provider_request_id"] == body["responseId"]
    assert provenance["provider_model_version"] == "gemini-3.8-flash"
    assert provenance["provider_metadata"]["http_status"] == 200
    assert provenance["provider_metadata"].get("finish_reason") == expected_finish_reason
    assert provenance["usage"]["input_tokens"] == body["usageMetadata"].get("promptTokenCount")
    assert provenance["usage"]["output_tokens"] == body["usageMetadata"].get("candidatesTokenCount")
    assert provenance["usage"]["reasoning_tokens"] == body["usageMetadata"].get(
        "thoughtsTokenCount"
    )
    assert len(session.calls) == 1


def test_openai_interpretation_error_also_preserves_diagnostics(monkeypatch):
    monkeypatch.setenv("TEST_OPENAI_KEY", "secret-key")
    body = {
        "id": "response-no-text",
        "model": "gpt-6-luna",
        "status": "incomplete",
        "output": [],
        "usage": {
            "input_tokens": 12,
            "output_tokens": 8,
            "output_tokens_details": {"reasoning_tokens": 8},
        },
    }
    session = FakeSession(FakeHTTPResponse(body))
    config = ProviderConfig.from_dict(
        {
            "provider": "openai",
            "model": "gpt-6-luna",
            "api_key_env": "TEST_OPENAI_KEY",
        }
    )
    analyzer = LLMAnalyzer(
        load_profile("reentrancy"), config, provider=OpenAIResponsesProvider(config, session)
    )
    provenance = analyzer.analyze("contract A {}", "A.sol", "zero_shot")["provenance"]
    assert json.loads(provenance["raw_provider_response"]) == body
    assert provenance["provider_request_id"] == "response-no-text"
    assert provenance["provider_metadata"] == {"http_status": 200, "status": "incomplete"}
    assert provenance["usage"] == {
        "input_tokens": 12,
        "output_tokens": 8,
        "reasoning_tokens": 8,
    }


def test_provenance_does_not_persist_secrets(monkeypatch):
    monkeypatch.setenv("UNUSED_IN_FAKE", "super-secret-key")
    provider = FakeProvider([valid_response()])
    config = provider_config(
        base_url="https://user:password@provider.example/v1?api_key=secret",
        options={"request_body": {"custom": True}, "authorization": "secret"},
    )
    analyzer = LLMAnalyzer(load_profile("reentrancy"), config, provider=provider)
    provenance = analyzer.analyze("contract A {}", "A.sol", "zero_shot")["provenance"]
    serialized = json.dumps(provenance)
    assert "super-secret-key" not in serialized
    assert "password" not in serialized
    assert "api_key=secret" not in serialized
    assert '"authorization": "secret"' not in serialized
    assert provenance["endpoint"] == "https://provider.example/v1"
    assert provenance["inference_parameters"]["option_names"] == ["authorization", "request_body"]


def test_provenance_retains_normalized_reasoning_effort():
    provider = FakeProvider([valid_response()])
    config = provider_config(
        provider="openai_compatible",
        base_url="https://provider.example/api",
        reasoning={"effort": "medium"},
    )
    analyzer = LLMAnalyzer(load_profile("reentrancy"), config, provider=provider)
    provenance = analyzer.analyze("contract A {}", "A.sol", "zero_shot")["provenance"]
    assert provenance["inference_parameters"]["reasoning"] == {"effort": "medium"}


def test_reentrant_finding_uses_deterministic_role_preference():
    raw = json.dumps(
        {
            "classification": "Reentrant",
            "vulnerable_functions": ["withdraw"],
            "reentered_functions": ["withdraw"],
            "reentrancy_type": "same_function",
            "relevant_lines": [
                {"line": 2, "role": "state_effect"},
                {"line": 3, "role": "reentry_target"},
                {"line": 4, "role": "external_interaction"},
            ],
            "explanation": "Exploit path.",
        }
    )
    analyzer = LLMAnalyzer(
        load_profile("reentrancy"), provider_config(), provider=FakeProvider([valid_response(raw)])
    )
    result = analyzer.analyze("a\nb\nc\nd\n", "A.sol", "zero_shot")
    assert result["findings"][0]["line"] == 4


def test_reentrant_finding_falls_back_through_remaining_role_preference():
    raw = json.dumps(
        {
            "classification": "Reentrant",
            "vulnerable_functions": ["withdraw"],
            "reentered_functions": ["transferBalance"],
            "reentrancy_type": "cross_function",
            "relevant_lines": [
                {"line": 1, "role": "other"},
                {"line": 2, "role": "state_effect"},
                {"line": 3, "role": "reentry_target"},
            ],
            "explanation": "Exploit path.",
        }
    )
    analyzer = LLMAnalyzer(
        load_profile("reentrancy"), provider_config(), provider=FakeProvider([valid_response(raw)])
    )
    result = analyzer.analyze("a\nb\nc\n", "A.sol", "zero_shot")
    assert result["findings"][0]["line"] == 3
