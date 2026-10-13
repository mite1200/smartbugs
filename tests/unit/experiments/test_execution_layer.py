import csv
import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from experiments.reentrancy.execution import (
    Condition,
    Contract,
    Observation,
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


def test_contract_id_selects_exactly_one_manifest_contract():
    contracts = [
        Contract("a.sol", "a.sol", "Safe", "hash-a"),
        Contract("nested/b.sol", "nested/b.sol", "Reentrant", "hash-b"),
    ]
    selected = RUNNER.select_contracts(contracts, None, "nested/b.sol")
    assert selected == [contracts[1]]


def test_unknown_contract_id_fails_clearly():
    contracts = [Contract("a.sol", "a.sol", "Safe", "hash-a")]
    with pytest.raises(
        ValueError, match="contract_id not found in canonical manifest: missing.sol"
    ):
        RUNNER.select_contracts(contracts, None, "missing.sol")


def test_contract_id_and_limit_are_mutually_exclusive():
    contracts = [Contract("a.sol", "a.sol", "Safe", "hash-a")]
    with pytest.raises(ValueError, match="mutually exclusive"):
        RUNNER.select_contracts(contracts, 1, "a.sol")


def pacing_observations(tmp_path: Path, counts: list[int]):
    condition = Condition(
        dataset="aggregated",
        model_condition="qwen3.8-27b-cerebras",
        model="qwen-3.8-27b",
        provider={},
        strategy="zero_shot",
        repeat=1,
        runid="run",
        rate_limit={
            "window_seconds": 60,
            "max_input_tokens_per_window": 120000,
            "max_requests_per_window": 32,
            "token_count_source": "tokens.csv",
        },
    )
    observations = []
    token_counts = {}
    for index, count in enumerate(counts):
        contract = Contract(f"contract-{index}", f"contract-{index}", "Safe", f"hash-{index}")
        observations.append(Observation(condition, contract, tmp_path / f"contract-{index}"))
        token_counts[(contract.contract_id, "zero_shot")] = count
    return condition, observations, token_counts


def test_pacing_windows_are_deterministic_and_enforce_token_limit(tmp_path):
    _, observations, token_counts = pacing_observations(tmp_path, [60000, 60000, 1])
    rate_limit = observations[0].condition.rate_limit
    first = RUNNER.pacing_windows(observations, "zero_shot", rate_limit, token_counts)
    second = RUNNER.pacing_windows(observations, "zero_shot", rate_limit, token_counts)
    assert first == second
    assert [window.input_tokens for window in first] == [120000, 1]
    assert [len(window.observations) for window in first] == [2, 1]
    assert all(window.input_tokens <= 120000 for window in first)


def test_pacing_windows_enforce_32_request_limit(tmp_path):
    _, observations, token_counts = pacing_observations(tmp_path, [1] * 33)
    windows = RUNNER.pacing_windows(
        observations, "zero_shot", observations[0].condition.rate_limit, token_counts
    )
    assert [len(window.observations) for window in windows] == [32, 1]


def test_large_observations_are_placed_without_reordering(tmp_path):
    _, observations, token_counts = pacing_observations(tmp_path, [70000, 50000, 70000])
    windows = RUNNER.pacing_windows(
        observations, "zero_shot", observations[0].condition.rate_limit, token_counts
    )
    assert [window.input_tokens for window in windows] == [120000, 70000]
    assert [
        [observation.contract.contract_id for observation in window.observations]
        for window in windows
    ] == [["contract-0", "contract-1"], ["contract-2"]]


def test_observation_exceeding_window_budget_fails(tmp_path):
    _, observations, token_counts = pacing_observations(tmp_path, [120001])
    with pytest.raises(ValueError, match="exceeding the per-window budget 120000"):
        RUNNER.pacing_windows(
            observations, "zero_shot", observations[0].condition.rate_limit, token_counts
        )


def test_missing_token_count_entry_fails(tmp_path):
    _, observations, _ = pacing_observations(tmp_path, [1])
    with pytest.raises(ValueError, match="missing token-count entry"):
        RUNNER.pacing_windows(observations, "zero_shot", observations[0].condition.rate_limit, {})


def test_duplicate_token_count_entry_fails(tmp_path):
    token_file = tmp_path / "tokens.csv"
    token_file.write_text(
        "contract_id,strategy,input_tokens\n" "a.sol,zero_shot,10\n" "a.sol,zero_shot,11\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate token-count entry"):
        RUNNER.load_token_counts(token_file)


def test_terminal_observations_are_excluded_before_window_construction(tmp_path):
    experiment, first = observation_fixture(tmp_path)
    condition = Condition(
        **{
            **first.condition.__dict__,
            "rate_limit": {
                "window_seconds": 60,
                "max_input_tokens_per_window": 120000,
                "max_requests_per_window": 32,
                "token_count_source": "tokens.csv",
            },
        }
    )
    first = Observation(condition, first.contract, first.result_directory)
    second_contract = Contract("second.sol", "second.sol", "Safe", "second-hash")
    second = Observation(condition, second_contract, tmp_path / "second.sol")
    write_provenance(first.result_directory, experiment, first, "invalid_output")
    missing = [
        observation
        for observation in (first, second)
        if inspect_observation(experiment, observation).state == "missing"
    ]
    windows = RUNNER.pacing_windows(
        missing,
        "zero_shot",
        condition.rate_limit,
        {("second.sol", "zero_shot"): 10},
    )
    assert [observation.contract.contract_id for observation in windows[0].observations] == [
        "second.sol"
    ]


def test_rate_limited_windows_wait_between_executed_windows(monkeypatch, tmp_path):
    condition, observations, token_counts = pacing_observations(tmp_path, [70000, 70000])
    windows = RUNNER.pacing_windows(observations, "zero_shot", condition.rate_limit, token_counts)
    calls = []
    sleeps = []
    monkeypatch.setattr(
        RUNNER, "refresh_window", lambda experiment, window: (window.observations, [])
    )
    RUNNER.execute_pacing_windows(
        tmp_path,
        tmp_path,
        {},
        condition,
        windows,
        4,
        execute=lambda *args: calls.append(args[4]),
        sleep=sleeps.append,
    )
    assert [len(call) for call in calls] == [1, 1]
    assert sleeps == [60.0]


def test_dry_run_returns_before_execution(monkeypatch, capsys, tmp_path):
    condition, observations, token_counts = pacing_observations(tmp_path, [10])
    monkeypatch.setattr(
        RUNNER,
        "plan_condition",
        lambda *args: ({}, condition, [], observations, []),
    )
    monkeypatch.setattr(
        RUNNER,
        "build_pacing_windows",
        lambda repository_root, selected_condition, missing: RUNNER.pacing_windows(
            missing, selected_condition.strategy, selected_condition.rate_limit, token_counts
        ),
    )
    monkeypatch.setattr(
        RUNNER,
        "execute_pacing_windows",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not execute")),
    )
    monkeypatch.setattr(
        RUNNER.sys,
        "argv",
        [
            "run_experiment.py",
            str(tmp_path),
            "--model-condition",
            condition.model_condition,
            "--strategy",
            condition.strategy,
            "--repeat",
            "1",
        ],
    )
    RUNNER.main()
    report = json.loads(capsys.readouterr().out)
    assert report["estimated_quota_windows"] == 1


def test_non_rate_limited_model_has_no_pacing_windows(tmp_path):
    _, conditions = configured_matrix()
    condition = next(item for item in conditions if item.model_condition == "gemini-3.8-flash")
    assert condition.rate_limit is None
    assert RUNNER.build_pacing_windows(tmp_path, condition, []) is None


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
