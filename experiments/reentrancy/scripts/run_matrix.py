#!/usr/bin/env python3
"""Plan or sequentially execute every configured experiment condition."""

import argparse
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.reentrancy.execution import (  # noqa: E402
    build_conditions,
    load_model_configs,
    load_yaml,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--limit-contracts", type=int)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    config_root = repository_root / "experiments/reentrancy/configs"
    experiment = load_yaml(config_root / "experiment.yaml")
    conditions = build_conditions(experiment, load_model_configs(config_root / "models"))
    runner = repository_root / "experiments/reentrancy/scripts/run_experiment.py"
    for condition in conditions:
        command = [
            sys.executable,
            str(runner),
            str(args.benchmark_root.resolve()),
            "--repository-root",
            str(repository_root),
            "--model-condition",
            condition.model_condition,
            "--strategy",
            condition.strategy,
            "--repeat",
            str(condition.repeat),
            "--processes",
            str(args.processes),
        ]
        if args.limit_contracts is not None:
            command.extend(["--limit-contracts", str(args.limit_contracts)])
        if args.execute:
            command.append("--execute")
        subprocess.run(command, check=True, cwd=repository_root)


if __name__ == "__main__":
    main()
