import json

from sb.llm.analyzer import LLMAnalyzer
from sb.llm.models import ProviderConfig, ProviderResponse
from sb.llm.profiles import load_profile
from sb.llm.providers import TransientProviderError


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
