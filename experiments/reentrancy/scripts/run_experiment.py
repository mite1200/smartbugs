#!/usr/bin/env python3
"""Plan or execute one deterministic reentrancy experiment condition."""

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import sb.settings  # noqa: E402
import sb.smartbugs  # noqa: E402

from experiments.reentrancy.execution import (  # noqa: E402
    Condition,
    ConditionLock,
    build_conditions,
    build_observations,
    inspect_observation,
    load_manifest,
    load_model_configs,
    load_yaml,
    verify_benchmark,
    verify_frozen_analyzer,
)


def select_condition(
    conditions: list[Condition], model_condition: str, strategy: str, repeat: int
) -> Condition:
    matches = [
        condition
        for condition in conditions
        if condition.model_condition == model_condition
        and condition.strategy == strategy
        and condition.repeat == repeat
    ]
    if len(matches) != 1:
        raise ValueError("exactly one experiment condition must match")
    return matches[0]


def select_contracts(contracts: list, limit: int | None, contract_id: str | None) -> list:
    if limit is not None and contract_id is not None:
        raise ValueError("--limit-contracts and --contract-id are mutually exclusive")
    if contract_id is not None:
        matches = [contract for contract in contracts if contract.contract_id == contract_id]
        if not matches:
            raise ValueError(f"contract_id not found in canonical manifest: {contract_id}")
        return matches
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        return contracts[:limit]
    return contracts


@dataclass(frozen=True)
class PacingWindow:
    observations: list
    input_tokens: int


def load_token_counts(path: Path) -> dict[tuple[str, str], int]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    counts: dict[tuple[str, str], int] = {}
    for row in rows:
        sample_id = row.get("contract_id") or row.get("scenario_id")
        if not sample_id:
            raise ValueError("token-count rows require contract_id or scenario_id")
        key = (sample_id, row["strategy"])
        if key in counts:
            raise ValueError(
                "duplicate token-count entry for " f"contract_id={key[0]!r}, strategy={key[1]!r}"
            )
        try:
            count = int(row["input_tokens"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid input_tokens for {key}") from error
        if count < 0:
            raise ValueError(f"negative input_tokens for {key}")
        counts[key] = count
    return counts


def pacing_windows(
    observations: list,
    strategy: str,
    rate_limit: dict[str, Any],
    token_counts: dict[tuple[str, str], int],
) -> list[PacingWindow]:
    max_tokens = int(rate_limit["max_input_tokens_per_window"])
    max_requests = int(rate_limit["max_requests_per_window"])
    if max_tokens <= 0 or max_requests <= 0:
        raise ValueError("rate-limit token and request budgets must be positive")
    windows: list[PacingWindow] = []
    current: list = []
    current_tokens = 0
    for observation in observations:
        key = (observation.contract.contract_id, strategy)
        if key not in token_counts:
            raise ValueError(
                "missing token-count entry for " f"contract_id={key[0]!r}, strategy={key[1]!r}"
            )
        tokens = token_counts[key]
        if tokens > max_tokens:
            raise ValueError(
                f"observation {key[0]!r}/{key[1]!r} requires {tokens} input tokens, "
                f"exceeding the per-window budget {max_tokens}"
            )
        if current and (len(current) >= max_requests or current_tokens + tokens > max_tokens):
            windows.append(PacingWindow(current, current_tokens))
            current = []
            current_tokens = 0
        current.append(observation)
        current_tokens += tokens
    if current:
        windows.append(PacingWindow(current, current_tokens))
    return windows


def build_pacing_windows(
    repository_root: Path, condition: Condition, missing: list
) -> list[PacingWindow] | None:
    if condition.rate_limit is None:
        return None
    source = condition.rate_limit.get("token_count_source")
    if not isinstance(source, str) or not source:
        raise ValueError("rate_limit.token_count_source must be a non-empty path")
    return pacing_windows(
        missing,
        condition.strategy,
        condition.rate_limit,
        load_token_counts(repository_root / source),
    )


def plan_condition(
    repository_root: Path,
    benchmark_root: Path,
    experiment_config: Path,
    model_condition: str,
    strategy: str,
    repeat: int,
    limit: int | None = None,
    contract_id: str | None = None,
) -> tuple[dict, Condition, list, list, list]:
    config_root = repository_root / "experiments/reentrancy/configs"
    experiment = load_yaml(experiment_config)
    contracts = load_manifest(repository_root / experiment["manifest"])
    contracts = select_contracts(contracts, limit, contract_id)
    verify_benchmark(benchmark_root, experiment, contracts)
    conditions = build_conditions(experiment, load_model_configs(config_root / "models"))
    condition = select_condition(conditions, model_condition, strategy, repeat)
    results_root = repository_root / experiment["results_root"]
    observations = build_observations([condition], contracts, results_root)
    completed, missing, conflicts = [], [], []
    for observation in observations:
        state = inspect_observation(experiment, observation)
        if state.state == "completed":
            completed.append((observation, state))
        elif state.state == "missing":
            missing.append(observation)
        else:
            conflicts.append((observation, state))
    return experiment, condition, completed, missing, conflicts


def smartbugs_configuration(
    repository_root: Path,
    benchmark_root: Path,
    experiment: dict,
    condition: Condition,
    missing: list,
    processes: int,
) -> dict:
    return {
        "tools": ["llm"],
        "files": [
            f"{benchmark_root}:{observation.contract.contract_path}" for observation in missing
        ],
        "json": True,
        "sarif": False,
        "processes": processes,
        "runid": condition.runid,
        "results": str(
            repository_root / experiment["results_root"] / "${RUNID}" / "${RELDIR}" / "${FILENAME}"
        ),
        "log": str(repository_root / experiment["logs_root"] / "${RUNID}.log"),
        "overwrite": False,
        "continue_on_errors": False,
        "llm": {
            "profile": experiment["profile"],
            "strategy": condition.strategy,
            "provider": condition.provider,
            "metadata": {
                "experiment_id": experiment["experiment_id"],
                "dataset": condition.dataset,
                "benchmark_commit": experiment["benchmark_commit"],
                "analyzer_tag": experiment["analyzer_tag"],
                "repeat_index": condition.repeat,
                "model_condition": condition.model_condition,
                "prompt_strategy": condition.strategy,
            },
        },
    }


def summary(
    condition: Condition,
    completed: list,
    missing: list,
    conflicts: list,
    windows: list[PacingWindow] | None = None,
) -> dict:
    report = {
        "runid": condition.runid,
        "model_condition": condition.model_condition,
        "model": condition.model,
        "strategy": condition.strategy,
        "repeat": condition.repeat,
        "selected_observations": len(completed) + len(missing) + len(conflicts),
        "completed_observations": len(completed),
        "missing_observations": len(missing),
        "conflict_observations": len(conflicts),
        "scheduled_contract_ids": [observation.contract.contract_id for observation in missing],
        "conflicts": [
            {
                "contract_id": observation.contract.contract_id,
                "reason": state.reason,
            }
            for observation, state in conflicts
        ],
    }
    if windows is not None:
        report["estimated_quota_windows"] = len(windows)
        report["pacing_windows"] = [
            {
                "window": index,
                "observation_count": len(window.observations),
                "input_tokens": window.input_tokens,
                "contract_ids": [
                    observation.contract.contract_id for observation in window.observations
                ],
                "strategy": condition.strategy,
            }
            for index, window in enumerate(windows, 1)
        ]
    return report


def execute_window(
    repository_root: Path,
    benchmark_root: Path,
    experiment: dict,
    condition: Condition,
    observations: list,
    processes: int,
) -> None:
    settings = sb.settings.Settings()
    settings.update(
        smartbugs_configuration(
            repository_root,
            benchmark_root,
            experiment,
            condition,
            observations,
            processes,
        )
    )
    sb.smartbugs.main(settings)


def refresh_window(experiment: dict, window: PacingWindow) -> tuple[list, list]:
    missing, conflicts = [], []
    for observation in window.observations:
        state = inspect_observation(experiment, observation)
        if state.state == "missing":
            missing.append(observation)
        elif state.state == "conflict":
            conflicts.append((observation, state))
    return missing, conflicts


def execute_pacing_windows(
    repository_root: Path,
    benchmark_root: Path,
    experiment: dict,
    condition: Condition,
    windows: list[PacingWindow],
    processes: int,
    execute: Callable[[Path, Path, dict, Condition, list, int], None] = execute_window,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    window_seconds = float(condition.rate_limit["window_seconds"] if condition.rate_limit else 0)
    if window_seconds <= 0:
        raise ValueError("rate_limit.window_seconds must be positive")
    for index, window in enumerate(windows):
        missing, conflicts = refresh_window(experiment, window)
        if conflicts:
            raise RuntimeError("conflicting result directories must be resolved manually")
        if missing:
            execute(
                repository_root,
                benchmark_root,
                experiment,
                condition,
                missing,
                processes,
            )
        if missing and index < len(windows) - 1:
            # Requests are started progressively by SmartBugs workers, not all
            # at the instant this batch begins. A full cooldown after the batch
            # completes guarantees that late requests cannot overlap the next
            # quota window.
            sleep(window_seconds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--model-condition", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--processes", type=int, default=1)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--limit-contracts", type=int)
    selection.add_argument("--contract-id")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("experiments/reentrancy/configs/experiment.yaml"),
    )
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    experiment, condition, completed, missing, conflicts = plan_condition(
        repository_root,
        benchmark_root,
        args.experiment_config.resolve(),
        args.model_condition,
        args.strategy,
        args.repeat,
        args.limit_contracts,
        args.contract_id,
    )
    windows = build_pacing_windows(repository_root, condition, missing)
    report = summary(condition, completed, missing, conflicts, windows)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not args.execute:
        return
    if args.processes <= 0:
        raise ValueError("processes must be positive")
    if conflicts:
        raise RuntimeError("conflicting result directories must be resolved manually")
    if not missing:
        return
    verify_frozen_analyzer(repository_root, experiment["analyzer_tag"])
    lock_path = repository_root / "experiments/reentrancy/runs/locks" / f"{condition.runid}.lock"
    with ConditionLock(lock_path):
        # Re-scan under the lock so concurrent or earlier invocations cannot
        # cause a paid observation to be repeated.
        experiment, condition, completed, missing, conflicts = plan_condition(
            repository_root,
            benchmark_root,
            args.experiment_config.resolve(),
            args.model_condition,
            args.strategy,
            args.repeat,
            args.limit_contracts,
            args.contract_id,
        )
        if conflicts:
            raise RuntimeError("conflicting result directories must be resolved manually")
        if not missing:
            return
        windows = build_pacing_windows(repository_root, condition, missing)
        if windows is None:
            execute_window(
                repository_root,
                benchmark_root,
                experiment,
                condition,
                missing,
                args.processes,
            )
        else:
            execute_pacing_windows(
                repository_root,
                benchmark_root,
                experiment,
                condition,
                windows,
                args.processes,
            )


if __name__ == "__main__":
    main()
