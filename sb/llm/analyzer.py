import random
import time
import uuid
from typing import Any, Callable, Optional

from sb.llm.models import ParseResult, ProviderConfig, ProviderResponse
from sb.llm.parser import parse_response
from sb.llm.preprocessing import preprocess
from sb.llm.profiles.base import AnalysisProfile
from sb.llm.providers import LLMProvider, TransientProviderError, create_provider
from sb.llm.provenance import build_provenance


class LLMAnalyzer:
    def __init__(
        self,
        profile: AnalysisProfile,
        provider_config: ProviderConfig,
        provider: Optional[LLMProvider] = None,
        sleep: Callable[[float], None] = time.sleep,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self.profile = profile
        self.provider_config = provider_config
        self.provider = provider or create_provider(provider_config)
        self.sleep = sleep
        self.jitter = jitter

    def analyze(
        self,
        source: str,
        contract_path: str,
        strategy: str,
        metadata: Optional[dict[str, Any]] = None,
        inference_id: Optional[str] = None,
    ) -> dict[str, Any]:
        run_id = inference_id or str(uuid.uuid4())
        processed = preprocess(source, self.profile.preprocessing)
        prompt = self.profile.render_prompt(strategy, processed)
        line_count = len(source.splitlines())
        retry_log: list[dict[str, Any]] = []
        started = time.monotonic()
        response: Optional[ProviderResponse] = None
        parse_result: Optional[ParseResult] = None
        technical_error: Optional[dict[str, Any]] = None
        try:
            response = self._infer(prompt, retry_log)
            parse_result = parse_response(response.text, self.profile, line_count)
        except Exception as error:
            technical_error = {"type": type(error).__name__, "message": str(error)}
        runtime = time.monotonic() - started
        provenance = build_provenance(
            inference_id=run_id,
            contract_path=contract_path,
            original_source=source,
            preprocessed_source=processed,
            profile=self.profile,
            strategy=strategy,
            provider_config=self.provider_config,
            rendered_prompt=prompt,
            provider_response=response,
            parse_result=parse_result,
            retry_log=retry_log,
            runtime_seconds=runtime,
            technical_error=technical_error,
            metadata=metadata or {},
        )
        return {
            "status": parse_result.status if parse_result else "technical_error",
            "parse_result": parse_result,
            "findings": self.profile.smartbugs_findings(parse_result) if parse_result else [],
            "provenance": provenance,
        }

    def _infer(self, prompt: str, retry_log: list[dict[str, Any]]) -> ProviderResponse:
        for attempt in range(self.provider_config.retry_count + 1):
            try:
                return self.provider.infer(prompt)
            except TransientProviderError as error:
                if attempt >= self.provider_config.retry_count:
                    raise
                delay = self.provider_config.retry_backoff_seconds * (2**attempt)
                delay += self.jitter() * self.provider_config.retry_backoff_seconds
                retry_log.append(
                    {
                        "attempt": attempt + 1,
                        "error_type": type(error).__name__,
                        "error": str(error),
                        "delay_seconds": delay,
                    }
                )
                self.sleep(delay)
        raise AssertionError("retry loop did not return or raise")
