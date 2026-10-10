import json
from typing import Optional

VERSION = "2026/10/10"
# profiles may map their result to arbitrary SmartBugs finding names
FINDINGS: set[str] = set()


def parse(
    exit_code: Optional[int], log: list[str], output: Optional[bytes]
) -> tuple[list[dict[str, object]], set[str], set[str], set[str]]:
    findings: list[dict[str, object]] = []
    infos: set[str] = set()
    errors: set[str] = set()
    fails: set[str] = set()
    if exit_code not in (0, None):
        fails.add(f"LLM_RUNNER_EXIT_CODE_{exit_code}")
    if output is None:
        fails.add("LLM runner produced no result")
        return findings, infos, errors, fails
    try:
        result = json.loads(output.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        fails.add(f"error parsing LLM runner result: {error}")
        return findings, infos, errors, fails
    findings.extend(result.get("findings") or [])
    parse_result = result.get("parse_result") or {}
    for warning in parse_result.get("warnings") or []:
        infos.add(f"LLM output warning: {warning}")
    if result.get("status") == "invalid_output":
        errors.add("LLM_INVALID_OUTPUT")
    technical_error = result.get("technical_error")
    if technical_error:
        error_type = technical_error.get("type", "error")
        message = technical_error.get("message", "")
        fails.add(f"LLM technical error ({error_type}: {message})")
    return findings, infos, errors, fails
