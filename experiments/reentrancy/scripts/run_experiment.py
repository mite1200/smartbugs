#!/usr/bin/env python3
"""Plan or execute one deterministic reentrancy experiment condition."""

import argparse
import json
import sys
from pathlib import Path

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


def plan_condition(
    repository_root: Path,
    benchmark_root: Path,
    model_condition: str,
    strategy: str,
    repeat: int,
    limit: int | None = None,
) -> tuple[dict, Condition, list, list, list]:
    config_root = repository_root / "experiments/reentrancy/configs"
    experiment = load_yaml(config_root / "experiment.yaml")
    contracts = load_manifest(repository_root / experiment["manifest"])
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        contracts = contracts[:limit]
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


def summary(condition: Condition, completed: list, missing: list, conflicts: list) -> dict:
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--model-condition", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--repeat", type=int, required=True)
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--limit-contracts", type=int)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    repository_root = args.repository_root.resolve()
    benchmark_root = args.benchmark_root.resolve()
    experiment, condition, completed, missing, conflicts = plan_condition(
        repository_root,
        benchmark_root,
        args.model_condition,
        args.strategy,
        args.repeat,
        args.limit_contracts,
    )
    report = summary(condition, completed, missing, conflicts)
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
            args.model_condition,
            args.strategy,
            args.repeat,
            args.limit_contracts,
        )
        if conflicts:
            raise RuntimeError("conflicting result directories must be resolved manually")
        if not missing:
            return
        settings = sb.settings.Settings()
        settings.update(
            smartbugs_configuration(
                repository_root,
                benchmark_root,
                experiment,
                condition,
                missing,
                args.processes,
            )
        )
        sb.smartbugs.main(settings)


if __name__ == "__main__":
    main()
