import csv
from pathlib import Path

ROOT = Path(__file__).parents[3]
EVALUATION = ROOT / "experiments/reentrancy/evaluation"

EXPECTED_F1 = {
    "aggregated": {
        ("gpt-6-luna", "zero_shot"): 0.9124,
        ("gpt-6-luna", "structured_reasoning"): 0.9099,
        ("gpt-6-luna", "few_shot"): 0.9108,
        ("gpt-6-luna", "structured_reasoning_few_shot"): 0.9231,
        ("grok-4.3", "zero_shot"): 0.9447,
        ("grok-4.3", "structured_reasoning"): 0.9258,
        ("grok-4.3", "few_shot"): 0.9356,
        ("grok-4.3", "structured_reasoning_few_shot"): 0.9397,
        ("qwen3.8-27b-cerebras", "zero_shot"): 0.9010,
        ("qwen3.8-27b-cerebras", "structured_reasoning"): 0.9109,
        ("qwen3.8-27b-cerebras", "few_shot"): 0.9202,
        ("qwen3.8-27b-cerebras", "structured_reasoning_few_shot"): 0.9048,
    },
    "rsd": {
        ("gpt-6-luna", "zero_shot"): 0.9143,
        ("gpt-6-luna", "structured_reasoning"): 0.9037,
        ("gpt-6-luna", "few_shot"): 0.9051,
        ("gpt-6-luna", "structured_reasoning_few_shot"): 0.9104,
        ("grok-4.3", "zero_shot"): 0.9130,
        ("grok-4.3", "structured_reasoning"): 0.8485,
        ("grok-4.3", "few_shot"): 0.8966,
        ("grok-4.3", "structured_reasoning_few_shot"): 0.9065,
        ("qwen3.8-27b-cerebras", "zero_shot"): 0.8043,
        ("qwen3.8-27b-cerebras", "structured_reasoning"): 0.8264,
        ("qwen3.8-27b-cerebras", "few_shot"): 0.8806,
        ("qwen3.8-27b-cerebras", "structured_reasoning_few_shot"): 0.9091,
    },
}


def read_csv(path: Path):
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def test_primary_metrics_match_exploratory_regression_values():
    for dataset, expected in EXPECTED_F1.items():
        rows = read_csv(EVALUATION / dataset / "metrics.csv")
        actual = {
            (row["model_condition"], row["prompt_strategy"]): round(float(row["f1"]), 4)
            for row in rows
        }
        assert actual == expected


def test_rsd_available_consensus_matched_regressions():
    rows = read_csv(EVALUATION / "rsd/within_model_matched_metrics.csv")
    qwen = {
        row["prompt_strategy"]: row
        for row in rows
        if row["model_condition"] == "qwen3.8-27b-cerebras"
    }
    assert {int(row["N"]) for row in qwen.values()} == {136}
    assert round(float(qwen["zero_shot"]["f1"]), 4) == 0.8387
    assert round(float(qwen["structured_reasoning"]["f1"]), 4) == 0.8346
    assert round(float(qwen["few_shot"]["f1"]), 4) == 0.8750
    assert round(float(qwen["structured_reasoning_few_shot"]["f1"]), 4) == 0.9147

    deltas = read_csv(EVALUATION / "rsd/prompt_deltas.csv")
    qwen_deltas = {
        row["prompt_strategy"]: round(float(row["f1_delta_vs_zero_shot"]), 4)
        for row in deltas
        if row["model_condition"] == "qwen3.8-27b-cerebras"
    }
    assert qwen_deltas["structured_reasoning"] == -0.0041
    assert qwen_deltas["few_shot"] == 0.0363
    assert qwen_deltas["structured_reasoning_few_shot"] == 0.0760

    fully = read_csv(EVALUATION / "rsd/fully_matched_metrics.csv")
    assert {int(row["N"]) for row in fully} == {135}


def test_aggregated_merge_preserves_both_experiment_ids_and_rsd_scenario_ids():
    aggregated = read_csv(EVALUATION / "aggregated/predictions.csv")
    assert {row["experiment_id"] for row in aggregated} == {
        "reentrancy-aggregated-main-v2",
        "reentrancy-aggregated-combined-v1",
    }
    assert len(aggregated) == 15552

    rsd = read_csv(EVALUATION / "rsd/predictions.csv")
    assert len(rsd) == 5148
    assert all(row["scenario_id"] == row["sample_id"] for row in rsd)
    assert all(row["scenario_id"].startswith("benchmarks/RSD/src/") for row in rsd)
