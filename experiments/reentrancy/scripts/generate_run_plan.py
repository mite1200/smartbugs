#!/usr/bin/env python3
"""Generate a deterministic Aggregated Benchmark run plan."""

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.reentrancy.execution import (  # noqa: E402
    build_conditions,
    load_manifest,
    load_model_configs,
    load_yaml,
    manifest_sha256,
)


def generate(repository_root: Path, experiment_config: Path, output: Path) -> dict:
    config_root = repository_root / "experiments/reentrancy/configs"
    experiment = load_yaml(experiment_config)
    manifest_path = repository_root / experiment["manifest"]
    contracts = load_manifest(manifest_path)
    conditions = build_conditions(experiment, load_model_configs(config_root / "models"))
    plan = {
        "experiment_id": experiment["experiment_id"],
        "dataset": experiment["dataset"],
        "dataset_name": experiment["dataset_name"],
        "benchmark_commit": experiment["benchmark_commit"],
        "analyzer_tag": experiment["analyzer_tag"],
        "manifest": experiment["manifest"],
        "manifest_sha256": manifest_sha256(manifest_path),
        "results_root": experiment["results_root"],
        "logs_root": experiment["logs_root"],
        "contract_count": len(contracts),
        "condition_count": len(conditions),
        "expected_observations": len(contracts) * len(conditions),
        "conditions": [
            {
                "dataset": condition.dataset,
                "model_condition": condition.model_condition,
                "model": condition.model,
                "provider": condition.provider["provider"],
                "strategy": condition.strategy,
                "repeat": condition.repeat,
                "runid": condition.runid,
                "expected_observations": len(contracts),
            }
            for condition in conditions
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--experiment-config",
        type=Path,
        default=Path("experiments/reentrancy/configs/experiment.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/reentrancy/runs/aggregated-plan.json"),
    )
    args = parser.parse_args()
    generate(
        args.repository_root.resolve(),
        args.experiment_config.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
