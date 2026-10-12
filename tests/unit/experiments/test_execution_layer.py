import csv
import importlib.util
import json
from pathlib import Path

import yaml

from experiments.reentrancy.execution import (
    Contract,
    build_conditions,
    build_observations,
    inspect_observation,
    load_model_configs,
    load_yaml,
    make_runid,
)

ROOT = Path(__file__).parents[3]
COLLECTOR_PATH = ROOT / "experiments/reentrancy/scripts/collect_predictions.py"
SPEC = importlib.util.spec_from_file_location("collect_predictions", COLLECTOR_PATH)
COLLECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COLLECTOR)

RUNNER_PATH = ROOT / "experiments/reentrancy/scripts/run_experiment.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_experiment", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def configured_matrix():
    config_root = ROOT / "experiments/reentrancy/configs"
    experiment = load_yaml(config_root / "experiment.yaml")
    models = load_model_configs(config_root / "models")
    return experiment, build_conditions(experiment, models)


def test_plan_has_27_deterministic_conditions_and_11664_observations():
    experiment, first = configured_matrix()
    _, second = configured_matrix()
    assert first == second
    assert len(first) == 27
    assert len({condition.runid for condition in first}) == 27
    assert len(first) * 432 == 11664
    assert (
        make_runid("aggregated", "gemini-3.8-flash", "zero_shot", 1)
        == "aggregated__gemini-3.8-flash__zero-shot__r1"
    )
    assert experiment["benchmark_commit"] == "d7c9496121931cff7c55b2047d18fb048fefe2cb"


def observation_fixture(tmp_path: Path, strategy: str = "zero_shot"):
    experiment, conditions = configured_matrix()
    condition = next(
        condition
        for condition in conditions
        if condition.model_condition == "gemini-3.8-flash"
        and condition.strategy == strategy
        and condition.repeat == 1
    )
    contract = Contract(
        contract_id="benchmarks/aggregated-benchmark/src/safe/example.sol",
        contract_path="benchmarks/aggregated-benchmark/src/safe/example.sol",
        ground_truth="Safe",
        sha256="source-hash",
    )
    observation = build_observations([condition], [contract], tmp_path)[0]
    return experiment, observation


def write_provenance(path: Path, experiment: dict, observation, status: str) -> None:
    path.mkdir(parents=True)
    technical_error = None
    parsed_result = {
        "status": status,
        "classification": "Safe" if status == "completed" else None,
    }
    if status == "technical_error":
        technical_error = {"type": "ProviderError", "message": "failed"}
        parsed_result = None
    provenance = {
        "inference_id": f"id-{status}",
        "contract_path": observation.contract.contract_path,
        "source_sha256": observation.contract.sha256,
        "preprocessed_source_sha256": "processed-hash",
        "model": observation.condition.model,
        "prompt_strategy": observation.condition.strategy,
        "parsed_result": parsed_result,
        "primary_classification": "Safe" if status == "completed" else None,
        "classification_valid": status == "completed",
        "schema_valid": status == "completed",
        "warnings": ["warning"] if status == "invalid_output" else [],
        "api_retry_count": 1,
        "runtime_seconds": 2.5,
        "technical_error": technical_error,
        "git_commit": "analyzer-commit",
        "provider_request_id": "request-id",
        "provider_model_version": "model-version",
        "usage": {"input_tokens": 10, "output_tokens": 5, "reasoning_tokens": 2},
        "metadata": {
            "experiment_id": experiment["experiment_id"],
            "dataset": observation.condition.dataset,
            "benchmark_commit": experiment["benchmark_commit"],
            "analyzer_tag": experiment["analyzer_tag"],
            "repeat_index": observation.condition.repeat,
            "model_condition": observation.condition.model_condition,
            "prompt_strategy": observation.condition.strategy,
        },
    }
    (path / "llm-provenance.json").write_text(json.dumps(provenance), encoding="utf-8")


def test_resume_skips_completed_valid_and_invalid_outputs(tmp_path):
    for status in ("completed", "invalid_output"):
        experiment, observation = observation_fixture(tmp_path / status)
        write_provenance(observation.result_directory, experiment, observation, status)
        state = inspect_observation(experiment, observation)
        assert state.state == "completed"
        assert state.status == status


def test_missing_is_scheduled_and_existing_partial_directory_is_blocked(tmp_path):
    experiment, observation = observation_fixture(tmp_path)
    assert inspect_observation(experiment, observation).state == "missing"
    observation.result_directory.mkdir(parents=True)
    (observation.result_directory / "result.log").write_text("partial", encoding="utf-8")
    state = inspect_observation(experiment, observation)
    assert state.state == "conflict"
    assert "without matching provenance" in state.reason


def test_smartbugs_configuration_never_overwrites_completed_work(tmp_path):
    experiment, observation = observation_fixture(tmp_path)
    configuration = RUNNER.smartbugs_configuration(
        ROOT,
        tmp_path / "benchmark",
        experiment,
        observation.condition,
        [observation],
        1,
    )
    assert configuration["overwrite"] is False
    assert configuration["files"] == [
        f"{tmp_path / 'benchmark'}:{observation.contract.contract_path}"
    ]
    assert configuration["llm"]["metadata"]["repeat_index"] == 1


def create_repository_fixture(tmp_path: Path):
    repository = tmp_path / "repo"
    config_root = repository / "experiments/reentrancy/configs"
    models_root = config_root / "models"
    manifests_root = repository / "experiments/reentrancy/manifests"
    models_root.mkdir(parents=True)
    manifests_root.mkdir(parents=True)
    experiment = {
        "experiment_id": "test-experiment",
        "dataset": "aggregated",
        "dataset_name": "Aggregated Benchmark",
        "manifest": "experiments/reentrancy/manifests/aggregated.csv",
        "benchmark_commit": "benchmark-commit",
        "analyzer_tag": "analyzer-tag",
        "profile": "reentrancy",
        "strategies": ["zero_shot"],
        "repeats": [1],
        "results_root": "experiments/reentrancy/runs/results",
        "logs_root": "experiments/reentrancy/runs/logs",
    }
    (config_root / "experiment.yaml").write_text(yaml.safe_dump(experiment), encoding="utf-8")
    model = {
        "condition_id": "model-a",
        "provider": {
            "provider": "openai_compatible",
            "model": "model-a-id",
            "base_url": "https://example.invalid",
            "api_key_env": "KEY",
        },
    }
    (models_root / "model-a.yaml").write_text(yaml.safe_dump(model), encoding="utf-8")
    rows = [
        {
            "contract_id": "benchmarks/aggregated-benchmark/src/safe/a.sol",
            "contract_path": "benchmarks/aggregated-benchmark/src/safe/a.sol",
            "ground_truth": "Safe",
            "dataset": "Aggregated Benchmark",
            "source": "unspecified",
            "sha256": "hash-a",
            "solidity_pragma": "",
            "solidity_version_family": "(no pragma)",
        },
        {
            "contract_id": "benchmarks/aggregated-benchmark/src/reentrant/b.sol",
            "contract_path": "benchmarks/aggregated-benchmark/src/reentrant/b.sol",
            "ground_truth": "Reentrant",
            "dataset": "Aggregated Benchmark",
            "source": "unspecified",
            "sha256": "hash-b",
            "solidity_pragma": "",
            "solidity_version_family": "(no pragma)",
        },
    ]
    with (manifests_root / "aggregated.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    return repository, experiment, rows


def test_collector_joins_ground_truth_and_reports_matrix_completeness(tmp_path):
    repository, experiment, rows = create_repository_fixture(tmp_path)
    config_root = repository / "experiments/reentrancy/configs"
    conditions = build_conditions(
        load_yaml(config_root / "experiment.yaml"), load_model_configs(config_root / "models")
    )
    contracts = [
        Contract(row["contract_id"], row["contract_path"], row["ground_truth"], row["sha256"])
        for row in rows
    ]
    observations = build_observations(
        conditions,
        contracts,
        repository / experiment["results_root"],
    )
    write_provenance(observations[0].result_directory, experiment, observations[0], "completed")
    collected, summary = COLLECTOR.collect(repository)
    assert [row["ground_truth"] for row in collected] == ["Reentrant", "Safe"]
    assert summary == {
        "experiment_id": "test-experiment",
        "dataset": "aggregated",
        "expected_observations": 2,
        "completed_observations": 1,
        "valid_completed_observations": 1,
        "missing_observations": 1,
        "invalid_output_observations": 0,
        "technical_failure_observations": 0,
        "incomplete_or_conflict_observations": 0,
        "duplicate_observations": 0,
        "duplicate_inference_ids": [],
        "condition_count": 1,
        "contract_count": 2,
    }
    completed = next(row for row in collected if row["status"] == "completed")
    assert completed["prediction"] == "Safe"
    assert completed["analyzer_tag"] == "analyzer-tag"


def test_collector_reports_invalid_and_technical_observations(tmp_path):
    repository, experiment, rows = create_repository_fixture(tmp_path)
    config_root = repository / "experiments/reentrancy/configs"
    conditions = build_conditions(
        load_yaml(config_root / "experiment.yaml"), load_model_configs(config_root / "models")
    )
    contracts = [
        Contract(row["contract_id"], row["contract_path"], row["ground_truth"], row["sha256"])
        for row in rows
    ]
    observations = build_observations(
        conditions,
        contracts,
        repository / experiment["results_root"],
    )
    write_provenance(
        observations[0].result_directory, experiment, observations[0], "invalid_output"
    )
    write_provenance(
        observations[1].result_directory, experiment, observations[1], "technical_error"
    )
    collected, summary = COLLECTOR.collect(repository)
    assert [row["status"] for row in collected] == ["technical_error", "invalid_output"]
    assert summary["completed_observations"] == 2
    assert summary["valid_completed_observations"] == 0
    assert summary["invalid_output_observations"] == 1
    assert summary["technical_failure_observations"] == 1
    assert summary["missing_observations"] == 0


def test_collector_reports_duplicate_inference_ids(tmp_path):
    repository, experiment, rows = create_repository_fixture(tmp_path)
    config_root = repository / "experiments/reentrancy/configs"
    conditions = build_conditions(
        load_yaml(config_root / "experiment.yaml"), load_model_configs(config_root / "models")
    )
    contracts = [
        Contract(row["contract_id"], row["contract_path"], row["ground_truth"], row["sha256"])
        for row in rows
    ]
    observations = build_observations(
        conditions,
        contracts,
        repository / experiment["results_root"],
    )
    for observation in observations:
        write_provenance(observation.result_directory, experiment, observation, "completed")
        provenance_path = observation.result_directory / "llm-provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["inference_id"] = "duplicate-id"
        provenance_path.write_text(json.dumps(provenance), encoding="utf-8")
    _, summary = COLLECTOR.collect(repository)
    assert summary["duplicate_observations"] == 1
    assert summary["duplicate_inference_ids"] == ["duplicate-id"]
