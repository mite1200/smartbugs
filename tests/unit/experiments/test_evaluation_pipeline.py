import json
import importlib.util
from pathlib import Path

import pytest

from experiments.reentrancy.evaluation_pipeline import (
    add_unique_key,
    aggregate_predictions,
    binary_classification,
    classification_metrics,
    matched_subset_metrics,
    normalize_binary_label,
    prompt_delta_rows,
)

ROOT = Path(__file__).parents[3]
RUN_EVALUATION_PATH = ROOT / "experiments/reentrancy/scripts/run_evaluation.py"
RUN_EVALUATION_SPEC = importlib.util.spec_from_file_location(
    "run_evaluation_summary", RUN_EVALUATION_PATH
)
RUN_EVALUATION = importlib.util.module_from_spec(RUN_EVALUATION_SPEC)
RUN_EVALUATION_SPEC.loader.exec_module(RUN_EVALUATION)


def observation(
    sample_id: str,
    repeat: int,
    prediction: str = "Reentrant",
    *,
    valid: bool = True,
    model: str = "model-a",
    strategy: str = "zero_shot",
    truth: str = "Reentrant",
    dataset: str = "test",
):
    return {
        "dataset": dataset,
        "model_condition": model,
        "prompt_strategy": strategy,
        "sample_id": sample_id,
        "repeat_index": repeat,
        "ground_truth": truth,
        "contract_id": sample_id,
        "scenario_id": "",
        "prediction": prediction if valid else "",
        "binary_valid": valid,
    }


def test_binary_validity_uses_only_locked_primary_semantics():
    base = {
        "technical_error": None,
        "parsed_result": {"status": "completed", "classification": "  reENTRant  "},
        "classification_valid": True,
        "schema_valid": False,
    }
    assert binary_classification(base) == ("Reentrant", True)
    assert normalize_binary_label(" vulnerable ") is None

    technical = {**base, "technical_error": {"type": "ProviderError"}}
    assert binary_classification(technical)[1] is False
    invalid_classification = {**base, "classification_valid": False}
    assert binary_classification(invalid_classification)[1] is False
    invalid_status = {
        **base,
        "parsed_result": {"status": "invalid_output", "classification": "Safe"},
    }
    assert binary_classification(invalid_status)[1] is False


def test_strict_aggregation_requires_exactly_three_valid_and_accepts_two_one_majority():
    unanimous = [observation("a", repeat) for repeat in (1, 2, 3)]
    split = [
        observation("b", 1),
        observation("b", 2),
        observation("b", 3, "Safe"),
    ]
    only_two = [
        observation("c", 1),
        observation("c", 2),
        observation("c", 3, valid=False),
    ]
    rows = aggregate_predictions(unanimous + split + only_two, "strict")
    by_id = {row["sample_id"]: row for row in rows}

    assert by_id["a"]["majority_prediction"] == "Reentrant"
    assert by_id["a"]["agreement"] == "3/3"
    assert by_id["b"]["majority_prediction"] == "Reentrant"
    assert by_id["b"]["agreement"] == "2/3"
    assert by_id["c"]["aggregate_status"] == "incomplete"
    assert by_id["c"]["majority_prediction"] == ""


def test_available_consensus_accepts_two_agreeing_and_rejects_disagreement_or_one_valid():
    agreeing = [
        observation("agree", 1, "Safe", truth="Safe"),
        observation("agree", 2, "Safe", truth="Safe"),
        observation("agree", 3, valid=False, truth="Safe"),
    ]
    disagreeing = [
        observation("disagree", 1),
        observation("disagree", 2, "Safe"),
        observation("disagree", 3, valid=False),
    ]
    one_valid = [
        observation("one", 1),
        observation("one", 2, valid=False),
        observation("one", 3, valid=False),
    ]
    rows = aggregate_predictions(agreeing + disagreeing + one_valid, "available")
    by_id = {row["sample_id"]: row for row in rows}

    assert by_id["agree"]["majority_prediction"] == "Safe"
    assert by_id["agree"]["aggregate_source"] == "2_valid_agreement"
    assert by_id["disagree"]["aggregate_status"] == "incomplete"
    assert by_id["disagree"]["agreement"] == "1/1"
    assert by_id["one"]["aggregate_status"] == "incomplete"


def test_metric_formulas_include_mcc_and_safe_zero_denominators():
    rows = [
        {"ground_truth": "Reentrant", "majority_prediction": "Reentrant"},
        {"ground_truth": "Reentrant", "majority_prediction": "Safe"},
        {"ground_truth": "Safe", "majority_prediction": "Reentrant"},
        {"ground_truth": "Safe", "majority_prediction": "Safe"},
    ]
    result = classification_metrics(rows)
    assert result == {
        "N": 4,
        "TP": 1,
        "TN": 1,
        "FP": 1,
        "FN": 1,
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
        "mcc": 0.0,
        "accuracy": 0.5,
    }
    assert classification_metrics([])["mcc"] == 0.0


def available_row(sample_id, model, strategy, prediction="Reentrant", truth="Reentrant"):
    return {
        "dataset": "rsd",
        "sample_id": sample_id,
        "model_condition": model,
        "prompt_strategy": strategy,
        "aggregate_status": "complete",
        "majority_prediction": prediction,
        "ground_truth": truth,
    }


def test_matched_subset_intersections_and_prompt_deltas():
    rows = []
    for model in ("m1", "m2"):
        for strategy in ("zero_shot", "few_shot"):
            rows.append(available_row("shared", model, strategy))
    rows.extend(
        [
            available_row("m1-only", "m1", "zero_shot"),
            available_row("m1-only", "m1", "few_shot"),
            available_row("zero-only", "m1", "zero_shot"),
            available_row("zero-only", "m2", "zero_shot"),
        ]
    )

    within = matched_subset_metrics(rows, "within_model")
    assert {row["N"] for row in within if row["model_condition"] == "m1"} == {2}
    assert {row["N"] for row in within if row["model_condition"] == "m2"} == {1}
    cross = matched_subset_metrics(rows, "cross_model")
    assert {row["N"] for row in cross if row["prompt_strategy"] == "zero_shot"} == {2}
    assert {row["N"] for row in cross if row["prompt_strategy"] == "few_shot"} == {1}
    fully = matched_subset_metrics(rows, "fully_matched")
    assert {row["N"] for row in fully} == {1}

    deltas = prompt_delta_rows(within)
    zero = next(
        row
        for row in deltas
        if row["model_condition"] == "m1" and row["prompt_strategy"] == "zero_shot"
    )
    assert zero["f1_delta_vs_zero_shot"] == 0.0


def test_duplicate_repeat_structure_is_rejected():
    rows = [observation("a", 1), observation("a", 1), observation("a", 3)]
    with pytest.raises(ValueError, match="unexpected repeat structure"):
        aggregate_predictions(rows, "strict")


def test_duplicate_observation_key_is_rejected():
    keys = set()
    key = ("rsd", "model-a", "zero_shot", "scenario-a", 1)
    add_unique_key(keys, key)
    with pytest.raises(ValueError, match="duplicate observation key"):
        add_unique_key(keys, key)


def test_json_output_fixture_is_deterministically_ordered(tmp_path: Path):
    rows = [observation("b", repeat, model="m2") for repeat in (1, 2, 3)] + [
        observation("a", repeat, model="m1") for repeat in (1, 2, 3)
    ]
    first = aggregate_predictions(rows, "strict")
    second = aggregate_predictions(list(reversed(rows)), "strict")
    assert first == second
    path = tmp_path / "rows.json"
    path.write_text(json.dumps(first, sort_keys=True), encoding="utf-8")
    assert json.loads(path.read_text(encoding="utf-8")) == first


def test_console_summary_distinguishes_expected_complete_and_incomplete_aggregates():
    result = {
        "strict": [
            {"aggregate_status": "complete"},
            {"aggregate_status": "complete"},
            {"aggregate_status": "incomplete"},
        ]
    }
    assert RUN_EVALUATION.aggregate_summary(result) == {
        "expected_aggregates": 3,
        "complete_strict_aggregates": 2,
        "incomplete_aggregates": 1,
    }
