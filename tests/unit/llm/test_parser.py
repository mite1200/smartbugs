import json

import pytest

from sb.llm.parser import parse_response
from sb.llm.profiles import load_profile


@pytest.fixture
def profile():
    return load_profile("reentrancy")


def response(**overrides):
    value = {
        "classification": "Reentrant",
        "vulnerable_functions": ["withdraw"],
        "reentered_functions": ["withdraw"],
        "reentrancy_type": "same_function",
        "relevant_lines": [{"line": 3, "role": "external_interaction"}],
        "explanation": "A harmful path exists.",
    }
    value.update(overrides)
    return json.dumps(value)


def test_direct_json_and_label_normalization(profile):
    result = parse_response(response(classification=" reentrant "), profile, 10)
    assert result.classification == "Reentrant"
    assert result.classification_valid is True
    assert result.schema_valid is True
    assert result.raw_response == response(classification=" reentrant ")


def test_markdown_fence_and_surrounding_prose(profile):
    fenced = parse_response(f"```json\n{response()}\n```", profile, 10)
    prose = parse_response(f"Result follows. {response()} Thanks.", profile, 10)
    assert fenced.classification_valid is True
    assert prose.classification_valid is True
    assert fenced.warnings == ["stripped surrounding Markdown code fence"]
    assert "extracted first complete JSON object" in prose.warnings[0]


def test_malformed_auxiliary_field_keeps_primary_prediction(profile):
    result = parse_response(response(vulnerable_functions="withdraw"), profile, 10)
    assert result.classification_valid is True
    assert result.schema_valid is False
    assert result.classification == "Reentrant"


def test_unexpected_type_and_invalid_lines_warn_without_losing_classification(profile):
    result = parse_response(
        response(
            reentrancy_type="unknown",
            relevant_lines=[
                {"line": "3", "role": "external_interaction"},
                {"line": 99, "role": "state_effect"},
                {"line": 4, "role": "bad"},
                {"line": 5, "role": "other"},
            ],
        ),
        profile,
        10,
    )
    assert result.classification_valid is True
    assert result.schema_valid is False
    assert result.validated_response["relevant_lines"] == [{"line": 5, "role": "other"}]
    assert len(result.warnings) == 4


@pytest.mark.parametrize(
    "raw",
    [
        response(classification="probably vulnerable"),
        json.dumps({"vulnerable_functions": []}),
        "not json",
        '{"classification":"Safe","classification":"Reentrant"}',
    ],
)
def test_invalid_or_ambiguous_classification(profile, raw):
    result = parse_response(raw, profile, 10)
    assert result.status == "invalid_output"
    assert result.classification_valid is False
