import json
from typing import Any, Optional

from sb.llm.models import ParseResult
from sb.llm.profiles.base import AnalysisProfile


class JSONObject(dict[str, Any]):
    def __init__(self, pairs: list[tuple[str, Any]]) -> None:
        super().__init__(pairs)
        self.pairs = pairs


def parse_response(
    raw_response: str, profile: AnalysisProfile, source_line_count: int
) -> ParseResult:
    warnings: list[str] = []
    parsed = _parse_json(raw_response)
    if parsed is None:
        cleaned = _strip_code_fence(raw_response)
        if cleaned != raw_response.strip():
            warnings.append("stripped surrounding Markdown code fence")
        parsed = _parse_json(cleaned)
    if parsed is None:
        candidate = _first_json_object(raw_response)
        if candidate is not None:
            warnings.append("extracted first complete JSON object from surrounding text")
            parsed = _parse_json(candidate)
    if not isinstance(parsed, JSONObject):
        return ParseResult(
            status="invalid_output",
            raw_response=raw_response,
            parsed_response=None,
            validated_response=None,
            classification=None,
            classification_valid=False,
            schema_valid=False,
            warnings=warnings + ["response is not a valid JSON object"],
        )

    field = profile.primary_classification_field
    values = [value for key, value in parsed.pairs if key == field]
    normalized = [_normalize_label(value, profile.accepted_labels) for value in values]
    accepted = [value for value in normalized if value is not None]
    distinct = set(accepted)
    classification = next(iter(distinct)) if len(distinct) == 1 else None
    classification_valid = len(values) >= 1 and len(accepted) == len(values) and len(distinct) == 1
    if len(values) > 1:
        warnings.append(f"duplicate primary field {field!r}")
    if not classification_valid:
        warnings.append(f"no unambiguous accepted value for primary field {field!r}")

    parsed_dict = dict(parsed)
    if classification_valid:
        parsed_dict[field] = classification
    validated, auxiliary_warnings = profile.validate_auxiliary(parsed_dict, source_line_count)
    warnings.extend(auxiliary_warnings)
    schema_valid = classification_valid and len(values) == 1 and not auxiliary_warnings
    return ParseResult(
        status="completed" if classification_valid else "invalid_output",
        raw_response=raw_response,
        parsed_response=parsed_dict,
        validated_response=validated,
        classification=classification,
        classification_valid=classification_valid,
        schema_valid=schema_valid,
        warnings=warnings,
    )


def _parse_json(value: str) -> Optional[Any]:
    try:
        return json.loads(value.strip(), object_pairs_hook=JSONObject)
    except (json.JSONDecodeError, TypeError):
        return None


def _strip_code_fence(value: str) -> str:
    stripped = value.strip()
    if not stripped.startswith("```") or not stripped.endswith("```"):
        return stripped
    first_newline = stripped.find("\n")
    if first_newline < 0:
        return stripped
    return stripped[first_newline + 1 : -3].strip()


def _first_json_object(value: str) -> Optional[str]:
    start = value.find("{")
    while start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(value)):
            char = value[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return value[start : index + 1]
        start = value.find("{", start + 1)
    return None


def _normalize_label(value: Any, accepted_labels: tuple[str, ...]) -> Optional[str]:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    matches = [label for label in accepted_labels if label.casefold() == normalized]
    return matches[0] if len(matches) == 1 else None
