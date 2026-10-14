import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from experiments.reentrancy.execution import (
    build_conditions,
    load_manifest,
    load_model_configs,
    load_yaml,
)
from sb.llm.preprocessing import preprocess
from sb.llm.profiles import load_profile
from sb.llm.profiles.reentrancy import COMMON_OUTPUT

ROOT = Path(__file__).parents[3]
BENCHMARK = Path("/tmp/reentrancy-detection-llms-rsd-prep")
CONFIG = ROOT / "experiments/reentrancy/configs/rsd-v1.yaml"
MANIFEST = ROOT / "experiments/reentrancy/benchmarks/rsd/rsd.csv"
TOKEN_COUNTS = ROOT / "experiments/reentrancy/benchmarks/rsd/qwen-prompt-token-counts.csv"
PLAN = ROOT / "experiments/reentrancy/runs/rsd-plan.json"
RUNNER_PATH = ROOT / "experiments/reentrancy/scripts/run_experiment.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_rsd_experiment", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)

MODELS = {"gpt-6-luna", "grok-4.3", "qwen3.8-27b-cerebras"}
STRATEGIES = {
    "zero_shot",
    "structured_reasoning",
    "few_shot",
    "structured_reasoning_few_shot",
}
PROTECTED_HASHES = {
    "experiments/reentrancy/configs/experiment.yaml": (
        "1e36c08b8df42788d4e274e95cbf7598f38181605eea57c483c405465deda069"
    ),
    "experiments/reentrancy/configs/combined-v1.yaml": (
        "8ff76c051489bc232fbd4edcc352975848c063ab70444a06a279d60581ba04c2"
    ),
    "experiments/reentrancy/runs/aggregated-plan.json": (
        "ac6192c57e6339ad75d4b4ed7f9a956cfcdaddd2c8d5e1a3e92032a040aa872e"
    ),
    "experiments/reentrancy/runs/combined-plan.json": (
        "849fce17031b9f7774fd5ef79f857b802ded54b78ba715d17ba80670ea49c8c9"
    ),
}


def configured_rsd_matrix():
    experiment = load_yaml(CONFIG)
    models = load_model_configs(ROOT / "experiments/reentrancy/configs/models")
    return experiment, build_conditions(experiment, models)


def test_rsd_config_resolves_complete_matrix_and_qwen_override():
    experiment, conditions = configured_rsd_matrix()

    assert experiment["experiment_id"] == "reentrancy-rsd-main-v1"
    assert experiment["dataset"] == "rsd"
    assert experiment["results_root"] == "experiments/reentrancy/runs/rsd-v1/results"
    assert experiment["logs_root"] == "experiments/reentrancy/runs/rsd-v1/logs"
    assert len(conditions) == 36
    assert {condition.model_condition for condition in conditions} == MODELS
    assert {condition.strategy for condition in conditions} == STRATEGIES
    assert {condition.repeat for condition in conditions} == {1, 2, 3}
    assert all(condition.runid.startswith("rsd__") for condition in conditions)
    assert all("aggregated" not in condition.runid for condition in conditions)

    qwen = [
        condition for condition in conditions if condition.model_condition == "qwen3.8-27b-cerebras"
    ]
    assert len(qwen) == 12
    assert all(
        condition.rate_limit["token_count_source"]
        == "experiments/reentrancy/benchmarks/rsd/qwen-prompt-token-counts.csv"
        for condition in qwen
    )


def test_rsd_plan_is_fully_crossed_and_contains_every_scenario_per_condition():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    contracts = load_manifest(MANIFEST)
    scenario_ids = {contract.contract_id for contract in contracts}
    labels = Counter(contract.ground_truth for contract in contracts)

    assert plan["experiment_id"] == "reentrancy-rsd-main-v1"
    assert plan["dataset"] == "rsd"
    assert plan["contract_count"] == 143
    assert plan["condition_count"] == 36
    assert plan["expected_observations"] == 5148
    assert labels == {"Reentrant": 71, "Safe": 72}
    assert {condition["model_condition"] for condition in plan["conditions"]} == MODELS
    assert {condition["strategy"] for condition in plan["conditions"]} == STRATEGIES
    assert {condition["repeat"] for condition in plan["conditions"]} == {1, 2, 3}
    for condition in plan["conditions"]:
        assert condition["expected_observations"] == 143
        assert set(condition["contract_ids"]) == scenario_ids
        assert all(
            identifier.startswith("benchmarks/RSD/src/") for identifier in condition["contract_ids"]
        )
        assert condition["runid"].startswith("rsd__")
        assert "aggregated" not in condition["runid"]


def test_rsd_qwen_token_counts_cover_all_scenarios_and_pace_every_strategy():
    contracts = load_manifest(MANIFEST)
    token_counts = RUNNER.load_token_counts(TOKEN_COUNTS)
    experiment, conditions = configured_rsd_matrix()
    qwen_conditions = [
        condition for condition in conditions if condition.model_condition == "qwen3.8-27b-cerebras"
    ]
    scenario_ids = {contract.contract_id for contract in contracts}

    assert len(token_counts) == 143 * 4
    assert {sample_id for sample_id, _ in token_counts} == scenario_ids
    assert {strategy for _, strategy in token_counts} == STRATEGIES

    results_root = ROOT / experiment["results_root"]
    windows_by_strategy = {}
    for condition in qwen_conditions:
        observations = RUNNER.build_observations([condition], contracts, results_root)
        windows = RUNNER.build_pacing_windows(ROOT, condition, observations)
        assert windows is not None
        assert sum(len(window.observations) for window in windows) == 143
        assert all(len(window.observations) <= 16 for window in windows)
        assert all(window.input_tokens <= 80000 for window in windows)
        windows_by_strategy.setdefault(condition.strategy, len(windows))
        assert windows_by_strategy[condition.strategy] == len(windows)

    assert set(windows_by_strategy) == STRATEGIES


def test_representative_rsd_prompt_keeps_active_multi_contract_source_together():
    source = (BENCHMARK / "benchmarks/RSD/src/09_ERC20Inherit_ree1.sol").read_text(encoding="utf-8")
    profile = load_profile("reentrancy")
    processed = preprocess(source, profile.preprocessing)

    assert "interface IERC20" in processed
    assert "contract MyERC20" in processed
    assert "contract C" in processed
    assert "dynamic dispatching" not in processed
    assert processed.startswith("001 | ")
    for strategy in STRATEGIES:
        prompt = profile.render_prompt(strategy, processed)
        assert prompt.endswith(processed)
        assert prompt.count(COMMON_OUTPUT) == 1
        assert "Reentrancy Scenarios Dataset" not in prompt


def test_completed_aggregated_configs_and_plans_are_byte_identical():
    for relative_path, expected_hash in PROTECTED_HASHES.items():
        assert hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest() == expected_hash


def test_rsd_manifest_and_token_artifact_have_exact_committed_counts():
    with MANIFEST.open(encoding="utf-8", newline="") as source:
        manifest_rows = list(csv.DictReader(source))
    with TOKEN_COUNTS.open(encoding="utf-8", newline="") as source:
        token_rows = list(csv.DictReader(source))

    assert len(manifest_rows) == 143
    assert Counter(row["ground_truth"] for row in manifest_rows) == {
        "Reentrant": 71,
        "Safe": 72,
    }
    assert len(token_rows) == 572
