from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class PreprocessingConfig:
    remove_comments: bool = False
    add_line_numbers: bool = False


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str
    api_key_env: str
    base_url: Optional[str] = None
    api_style: Optional[str] = None
    temperature: Optional[float] = None
    reasoning: dict[str, Any] = field(default_factory=dict)
    max_output_tokens: Optional[int] = None
    request_timeout: float = 60.0
    recommended_concurrency: Optional[int] = None
    retry_count: int = 2
    retry_backoff_seconds: float = 1.0
    options: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ProviderConfig":
        if not isinstance(value, dict):
            raise ValueError("provider configuration must be a mapping")
        provider = value.get("provider")
        model = value.get("model")
        api_key_env = value.get("api_key_env")
        if provider not in {"gemini", "openai", "openai_compatible"}:
            raise ValueError("provider must be gemini, openai, or openai_compatible")
        if not isinstance(model, str) or not model.strip():
            raise ValueError("provider model must be a non-empty string")
        if not isinstance(api_key_env, str) or not api_key_env.strip():
            raise ValueError("api_key_env must be a non-empty string")
        base_url = value.get("base_url")
        if provider == "openai_compatible" and not base_url:
            raise ValueError("base_url is required for openai_compatible providers")
        api_style = value.get("api_style")
        if provider == "openai":
            api_style = "responses"
        elif provider == "openai_compatible":
            api_style = api_style or "chat_completions"
            if api_style not in {"chat_completions", "responses"}:
                raise ValueError("api_style must be chat_completions or responses")
        reasoning = value.get("reasoning") or {}
        options = value.get("options") or {}
        if not isinstance(reasoning, dict):
            raise ValueError("reasoning must be a mapping")
        if not isinstance(options, dict):
            raise ValueError("provider options must be a mapping")
        recommended = value.get("recommended_concurrency")
        if recommended is not None:
            recommended = _positive_int(recommended, "recommended_concurrency")
        max_tokens = value.get("max_output_tokens")
        if max_tokens is not None:
            max_tokens = _positive_int(max_tokens, "max_output_tokens")
        temperature = value.get("temperature")
        if temperature is not None:
            temperature = float(temperature)
        _validate_provider_compatibility(
            provider=provider,
            model=model,
            temperature=temperature,
            reasoning=reasoning,
            options=options,
        )
        return cls(
            provider=provider,
            model=model,
            api_key_env=api_key_env,
            base_url=base_url,
            api_style=api_style,
            temperature=temperature,
            reasoning=reasoning,
            max_output_tokens=max_tokens,
            request_timeout=_positive_float(value.get("request_timeout", 60), "request_timeout"),
            recommended_concurrency=recommended,
            retry_count=_non_negative_int(value.get("retry_count", 2), "retry_count"),
            retry_backoff_seconds=_non_negative_float(
                value.get("retry_backoff_seconds", 1), "retry_backoff_seconds"
            ),
            options=options,
        )

    def public_dict(self) -> dict[str, Any]:
        """Return only configuration values safe to persist in provenance."""
        return {
            "provider": self.provider,
            "model": self.model,
            "api_key_env": self.api_key_env,
            "base_url": _safe_endpoint(self.base_url),
            "api_style": self.api_style,
            "temperature": self.temperature,
            "reasoning": self.reasoning,
            "max_output_tokens": self.max_output_tokens,
            "request_timeout": self.request_timeout,
            "recommended_concurrency": self.recommended_concurrency,
            "retry_count": self.retry_count,
            "retry_backoff_seconds": self.retry_backoff_seconds,
            # provider options can contain arbitrary endpoint-specific values;
            # they affect inference but are left out here to prevent accidental
            # persistence of credentials, cookies, or authentication data
            "option_names": sorted(self.options),
        }


@dataclass(frozen=True)
class ProviderResponse:
    raw_response: str
    text: str
    request_id: Optional[str] = None
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    reasoning_tokens: Optional[int] = None
    model_version: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseResult:
    status: str
    raw_response: str
    parsed_response: Optional[dict[str, Any]]
    validated_response: Optional[dict[str, Any]]
    classification: Optional[str]
    classification_valid: bool
    schema_valid: bool
    warnings: list[str]

    def dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "raw_response": self.raw_response,
            "parsed_response": self.parsed_response,
            "validated_response": self.validated_response,
            "classification": self.classification,
            "classification_valid": self.classification_valid,
            "schema_valid": self.schema_valid,
            "warnings": self.warnings,
        }


def _positive_int(value: Any, name: str) -> int:
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _non_negative_int(value: Any, name: str) -> int:
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if result <= 0:
        raise ValueError(f"{name} must be positive")
    return result


def _non_negative_float(value: Any, name: str) -> float:
    result = float(value)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def _safe_endpoint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def _validate_provider_compatibility(
    *,
    provider: str,
    model: str,
    temperature: Optional[float],
    reasoning: dict[str, Any],
    options: dict[str, Any],
) -> None:
    request_body = _option_mapping(options, "request_body")
    if provider == "gemini" and model == "gemini-3.8-flash":
        unsupported_reasoning = set(reasoning).difference({"effort"})
        if unsupported_reasoning:
            names = ", ".join(sorted(unsupported_reasoning))
            raise ValueError(
                f"gemini-3.8-flash reasoning supports only 'effort'; unsupported field(s): {names}"
            )
        effort = reasoning.get("effort")
        if effort is not None and effort not in {"low", "medium", "high"}:
            raise ValueError("gemini-3.8-flash reasoning.effort must be one of: low, medium, high")
        generation_config = _option_mapping(options, "generation_config")
        forbidden = []
        if temperature is not None:
            forbidden.append("temperature")
        for key in ("temperature", "top_p", "topP", "top_k", "topK"):
            if key in generation_config:
                forbidden.append(f"options.generation_config.{key}")
            if key in request_body:
                forbidden.append(f"options.request_body.{key}")
        if "thinkingConfig" in generation_config:
            forbidden.append("options.generation_config.thinkingConfig")
        if "generationConfig" in request_body:
            forbidden.append("options.request_body.generationConfig")
        if forbidden:
            raise ValueError(
                "gemini-3.8-flash does not support sampling configuration: " + ", ".join(forbidden)
            )

    effort = reasoning.get("effort")
    if provider == "openai" and "reasoning" in request_body:
        raise ValueError(
            "native OpenAI reasoning must be configured with the top-level reasoning mapping"
        )
    if provider == "openai" and effort is not None and effort != "none":
        forbidden = []
        if temperature is not None:
            forbidden.append("temperature")
        for key in ("temperature", "top_p"):
            if key in request_body:
                forbidden.append(f"options.request_body.{key}")
        if forbidden:
            raise ValueError(
                "native OpenAI reasoning with effort other than 'none' cannot be combined with: "
                + ", ".join(forbidden)
            )


def _option_mapping(options: dict[str, Any], key: str) -> dict[str, Any]:
    value = options.get(key) or {}
    if not isinstance(value, dict):
        raise ValueError(f"provider option {key!r} must be a mapping")
    return value
