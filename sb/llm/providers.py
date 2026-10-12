import json
import os
from abc import ABC, abstractmethod
from typing import Any, Optional
from urllib.parse import quote

import requests

from sb.llm.models import ProviderConfig, ProviderResponse


class ProviderError(Exception):
    def __init__(self, message: str, response: Optional[ProviderResponse] = None) -> None:
        super().__init__(message)
        self.response = response


class TransientProviderError(ProviderError):
    pass


class LLMProvider(ABC):
    def __init__(self, config: ProviderConfig, session: Optional[requests.Session] = None) -> None:
        self.config = config
        self.session = session or requests.Session()
        api_key = os.environ.get(config.api_key_env)
        if not api_key:
            raise ProviderError(f"environment variable {config.api_key_env} is not set or is empty")
        self.api_key = api_key

    @abstractmethod
    def infer(self, prompt: str) -> ProviderResponse:
        pass

    def _post(
        self, url: str, headers: dict[str, str], payload: dict[str, Any]
    ) -> requests.Response:
        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.config.request_timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as error:
            raise TransientProviderError(str(error)) from error
        if response.status_code == 429 or 500 <= response.status_code <= 599:
            raise TransientProviderError(
                f"provider returned HTTP {response.status_code}: {self._redact(response.text)}"
            )
        if response.status_code >= 400:
            raise ProviderError(
                f"provider returned HTTP {response.status_code}: {self._redact(response.text)}"
            )
        return response

    def _redact(self, value: str) -> str:
        return value.replace(self.api_key, "[REDACTED]")

    def _diagnostic_response(
        self,
        response: requests.Response,
        *,
        request_id: Optional[str] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        reasoning_tokens: Optional[int] = None,
        model_version: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ProviderResponse:
        diagnostic_metadata = dict(metadata or {})
        diagnostic_metadata["http_status"] = response.status_code
        return ProviderResponse(
            raw_response=self._redact(response.text),
            text=None,
            request_id=request_id or response.headers.get("x-request-id"),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=reasoning_tokens,
            model_version=model_version,
            metadata=diagnostic_metadata,
        )

    def _response_json(self, response: requests.Response) -> dict[str, Any]:
        try:
            return _response_json(response)
        except ProviderError as error:
            raise ProviderError(str(error), self._diagnostic_response(response)) from error


class OpenAIResponsesProvider(LLMProvider):
    def infer(self, prompt: str) -> ProviderResponse:
        base_url = (self.config.base_url or "https://api.openai.com/v1").rstrip("/")
        payload: dict[str, Any] = {"model": self.config.model, "input": prompt}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            payload["max_output_tokens"] = self.config.max_output_tokens
        if self.config.reasoning:
            payload["reasoning"] = self.config.reasoning
        payload.update(_mapping_option(self.config.options, "request_body"))
        # smartbugs performs independent single-turn analyses and retains its
        # own provenance anyway, so provider-side response storage is not requested
        payload["store"] = False
        response = self._post(
            f"{base_url}/responses",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            payload,
        )
        data = self._response_json(response)
        usage = data.get("usage") or {}
        output_details = usage.get("output_tokens_details") or {}
        diagnostic = self._diagnostic_response(
            response,
            request_id=data.get("id"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_tokens=output_details.get("reasoning_tokens"),
            model_version=data.get("model"),
            metadata={"status": data.get("status")},
        )
        try:
            text = _responses_text(data)
        except ProviderError as error:
            raise ProviderError(str(error), diagnostic) from error
        return ProviderResponse(
            raw_response=self._redact(response.text),
            text=self._redact(text),
            request_id=data.get("id") or response.headers.get("x-request-id"),
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            reasoning_tokens=output_details.get("reasoning_tokens"),
            model_version=data.get("model"),
            metadata={"status": data.get("status")},
        )


class OpenAIChatProvider(LLMProvider):
    def infer(self, prompt: str) -> ProviderResponse:
        base_url = str(self.config.base_url).rstrip("/")
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            payload["max_tokens"] = self.config.max_output_tokens
        payload.update(_mapping_option(self.config.options, "request_body"))
        effort = self.config.reasoning.get("effort")
        if effort is not None:
            payload["reasoning_effort"] = effort
        response = self._post(
            f"{base_url}/chat/completions",
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            payload,
        )
        data = self._response_json(response)
        usage = data.get("usage") or {}
        details = usage.get("completion_tokens_details") or {}
        diagnostic = self._diagnostic_response(
            response,
            request_id=data.get("id"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            model_version=data.get("model"),
        )
        try:
            text = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError(
                "OpenAI-compatible response contains no message content", diagnostic
            ) from error
        return ProviderResponse(
            raw_response=self._redact(response.text),
            text=self._redact(text),
            request_id=data.get("id") or response.headers.get("x-request-id"),
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            reasoning_tokens=details.get("reasoning_tokens"),
            model_version=data.get("model"),
        )


class GeminiProvider(LLMProvider):
    def infer(self, prompt: str) -> ProviderResponse:
        base_url = (
            self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        ).rstrip("/")
        generation_config: dict[str, Any] = {}
        if self.config.temperature is not None and self.config.model != "gemini-3.8-flash":
            generation_config["temperature"] = self.config.temperature
        if self.config.max_output_tokens is not None:
            generation_config["maxOutputTokens"] = self.config.max_output_tokens
        generation_config.update(_mapping_option(self.config.options, "generation_config"))
        if self.config.model == "gemini-3.8-flash":
            effort = self.config.reasoning.get("effort")
            if effort is not None:
                generation_config["thinkingConfig"] = {"thinkingLevel": effort}
        else:
            generation_config.update(self.config.reasoning)
        payload: dict[str, Any] = {"contents": [{"parts": [{"text": prompt}]}]}
        if generation_config:
            payload["generationConfig"] = generation_config
        payload.update(_mapping_option(self.config.options, "request_body"))
        model = quote(self.config.model.removeprefix("models/"), safe="")
        response = self._post(
            f"{base_url}/models/{model}:generateContent",
            {"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
            payload,
        )
        data = self._response_json(response)
        usage = data.get("usageMetadata") or {}
        candidates = data.get("candidates") or []
        first_candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
        metadata = {}
        if first_candidate.get("finishReason") is not None:
            metadata["finish_reason"] = first_candidate["finishReason"]
        diagnostic = self._diagnostic_response(
            response,
            request_id=data.get("responseId"),
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            reasoning_tokens=usage.get("thoughtsTokenCount"),
            model_version=data.get("modelVersion"),
            metadata=metadata,
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as error:
            raise ProviderError("Gemini response contains no candidate text", diagnostic) from error
        if not text:
            raise ProviderError("Gemini response contains empty candidate text", diagnostic)
        return ProviderResponse(
            raw_response=self._redact(response.text),
            text=self._redact(text),
            request_id=data.get("responseId") or response.headers.get("x-request-id"),
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            reasoning_tokens=usage.get("thoughtsTokenCount"),
            model_version=data.get("modelVersion"),
            metadata={"finish_reason": (data.get("candidates") or [{}])[0].get("finishReason")},
        )


def create_provider(
    config: ProviderConfig, session: Optional[requests.Session] = None
) -> LLMProvider:
    if config.provider == "gemini":
        return GeminiProvider(config, session)
    if config.provider == "openai" or config.api_style == "responses":
        return OpenAIResponsesProvider(config, session)
    return OpenAIChatProvider(config, session)


def _response_json(response: requests.Response) -> dict[str, Any]:
    try:
        value = json.loads(response.text)
    except json.JSONDecodeError as error:
        raise ProviderError("provider returned a non-JSON HTTP response") from error
    if not isinstance(value, dict):
        raise ProviderError("provider returned a JSON value that is not an object")
    return value


def _responses_text(data: dict[str, Any]) -> str:
    texts: list[str] = []
    for item in data.get("output") or []:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for content in item.get("content") or []:
            if isinstance(content, dict) and content.get("type") == "output_text":
                texts.append(content.get("text", ""))
    text = "".join(texts)
    if not text:
        raise ProviderError("OpenAI Responses response contains no output text")
    return text


def _mapping_option(options: dict[str, Any], key: str) -> dict[str, Any]:
    value = options.get(key) or {}
    if not isinstance(value, dict):
        raise ProviderError(f"provider option {key!r} must be a mapping")
    return value
