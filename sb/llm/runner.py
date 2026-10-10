import json
import os
from typing import TYPE_CHECKING, Any, Optional

import sb.cfg
import sb.errors
import sb.io

from sb.llm.analyzer import LLMAnalyzer
from sb.llm.models import ProviderConfig
from sb.llm.profiles import load_profile

if TYPE_CHECKING:
    import sb.tasks


PROVENANCE_FILE = "llm-provenance.json"


def execute(
    task: "sb.tasks.Task",
) -> tuple[Optional[int], list[str], Optional[bytes], list[str], dict[str, Any]]:
    provenance_path = os.path.join(task.rdir, PROVENANCE_FILE)
    try:
        os.remove(provenance_path)
    except FileNotFoundError:
        pass
    try:
        configuration = _configuration(task)
        profile = load_profile(configuration["profile"])
        strategy = configuration["strategy"]
        provider_config = ProviderConfig.from_dict(configuration["provider"])
        source = sb.io.read_text(task.absfn)
        analyzer = LLMAnalyzer(profile, provider_config)
        result = analyzer.analyze(
            source=source,
            contract_path=task.relfn,
            strategy=strategy,
            metadata=configuration.get("metadata") or {},
        )
        provenance = result["provenance"]
        sb.io.write_json(provenance_path, provenance)
        serializable = {
            "status": result["status"],
            "findings": result["findings"],
            "parse_result": result["parse_result"].dict() if result["parse_result"] else None,
            "technical_error": provenance["technical_error"],
            "provenance_file": PROVENANCE_FILE,
        }
        logs = [
            f"LLM inference {provenance['inference_id']} status={result['status']}",
            f"provider={provider_config.provider} model={provider_config.model}",
        ]
        return (
            0,
            logs,
            json.dumps(serializable).encode("utf-8"),
            [],
            {
                "provider": provider_config.provider,
                "model": provider_config.model,
                "profile": profile.id,
                "strategy": strategy,
                "provenance": PROVENANCE_FILE,
            },
        )
    except (KeyError, TypeError, ValueError) as error:
        raise sb.errors.SmartBugsError(f"Invalid LLM analyzer configuration: {error}") from error
    except sb.errors.SmartBugsError:
        raise
    except Exception as error:
        raise sb.errors.SmartBugsError(f"LLM analyzer failed: {error}") from error


def _configuration(task: "sb.tasks.Task") -> dict[str, Any]:
    configuration = dict(getattr(task.tool, "options", {}) or {})
    configuration.update(getattr(task.settings, "llm", {}) or {})
    if not configuration:
        raise ValueError("the llm setting is required")
    metadata = configuration.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a mapping")
    return configuration
