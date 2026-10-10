import hashlib
import os
import subprocess
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit

import sb.cfg

from sb.llm.models import ParseResult, ProviderConfig, ProviderResponse
from sb.llm.profiles.base import AnalysisProfile


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_provenance(
    *,
    inference_id: str,
    contract_path: str,
    original_source: str,
    preprocessed_source: str,
    profile: AnalysisProfile,
    strategy: str,
    provider_config: ProviderConfig,
    rendered_prompt: str,
    provider_response: Optional[ProviderResponse],
    parse_result: Optional[ParseResult],
    retry_log: list[dict[str, Any]],
    runtime_seconds: float,
    technical_error: Optional[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Build an allowlisted provenance record; request headers are never accepted."""
    response = provider_response
    return {
        "inference_id": inference_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "contract_path": contract_path,
        "source_sha256": sha256_text(original_source),
        "preprocessed_source_sha256": sha256_text(preprocessed_source),
        "analysis_profile": profile.id,
        "profile_version": profile.version,
        "prompt_strategy": strategy,
        "prompt_version": profile.prompt_versions[strategy],
        "provider": provider_config.provider,
        "model": provider_config.model,
        "endpoint": _safe_endpoint(provider_config.base_url),
        "inference_parameters": provider_config.public_dict(),
        "rendered_prompt": rendered_prompt,
        "raw_provider_response": response.raw_response if response else None,
        "raw_model_text": response.text if response else None,
        "parsed_result": parse_result.dict() if parse_result else None,
        "primary_classification": parse_result.classification if parse_result else None,
        "classification_valid": parse_result.classification_valid if parse_result else False,
        "schema_valid": parse_result.schema_valid if parse_result else False,
        "warnings": parse_result.warnings if parse_result else [],
        "api_retry_count": len(retry_log),
        "retry_log": retry_log,
        "technical_error": technical_error,
        "runtime_seconds": runtime_seconds,
        "usage": {
            "input_tokens": response.input_tokens if response else None,
            "output_tokens": response.output_tokens if response else None,
            "reasoning_tokens": response.reasoning_tokens if response else None,
        },
        "provider_request_id": response.request_id if response else None,
        "provider_model_version": response.model_version if response else None,
        "provider_metadata": _sanitize(response.metadata) if response else {},
        "smartbugs_version": sb.cfg.SB_VERSION,
        "git_commit": _git_commit(),
        "metadata": _sanitize(metadata),
    }


def _git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=sb.cfg.SB_HOME,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return os.environ.get("SMARTBUGS_GIT_COMMIT")


def _safe_endpoint(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    parsed = urlsplit(value)
    hostname = parsed.hostname or ""
    if parsed.port:
        hostname = f"{hostname}:{parsed.port}"
    return urlunsplit((parsed.scheme, hostname, parsed.path, "", ""))


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in {
                "api_key",
                "apikey",
                "authorization",
                "cookie",
                "cookies",
                "password",
                "secret",
                "access_token",
                "refresh_token",
            }:
                result[key] = "[REDACTED]"
            else:
                result[key] = _sanitize(item)
        return result
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value
