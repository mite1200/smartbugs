#!/usr/bin/env python3
"""Regenerate all thesis reentrancy LLM evaluation artifacts."""

import argparse
import json
import shutil
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.reentrancy.evaluation_pipeline import (  # noqa: E402
    evaluate_dataset,
    write_csv,
)


def aggregate_summary(result: dict) -> dict[str, int]:
    expected = len(result["strict"])
    complete = sum(row["aggregate_status"] == "complete" for row in result["strict"])
    return {
        "expected_aggregates": expected,
        "complete_strict_aggregates": complete,
        "incomplete_aggregates": expected - complete,
    }


def run(repository_root: Path) -> dict:
    output_root = repository_root / "experiments/reentrancy/evaluation"
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)
    config_root = repository_root / "experiments/reentrancy/configs"

    aggregated = evaluate_dataset(
        repository_root,
        [config_root / "experiment.yaml", config_root / "combined-v1.yaml"],
        output_root / "aggregated",
    )
    rsd = evaluate_dataset(
        repository_root,
        [config_root / "rsd-v1.yaml"],
        output_root / "rsd",
    )

    source_metrics = sorted(
        [*aggregated["metrics"], *rsd["metrics"]],
        key=lambda row: (row["dataset"], row["model_condition"], row["prompt_strategy"]),
    )
    combined_metrics = [
        {
            "dataset": row["dataset"],
            "model": row["model_condition"],
            "strategy": row["prompt_strategy"],
            "N": row["N"],
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1"],
            "mcc": row["mcc"],
            "accuracy": row["accuracy"],
            "complete": row["complete_samples"],
            "incomplete": row["incomplete_samples"],
        }
        for row in source_metrics
    ]
    combined_coverage = [
        {
            "dataset": row["dataset"],
            "model": row["model_condition"],
            "strategy": row["prompt_strategy"],
            "expected_samples": row["expected_samples"],
            "complete_samples": row["complete_samples"],
            "incomplete_samples": row["incomplete_samples"],
            "incomplete_reentrant": row["incomplete_reentrant"],
            "incomplete_safe": row["incomplete_safe"],
        }
        for row in source_metrics
    ]
    write_csv(output_root / "combined/metrics.csv", combined_metrics)
    write_csv(output_root / "combined/coverage.csv", combined_coverage)

    aggregated_summary = aggregate_summary(aggregated)
    rsd_summary = aggregate_summary(rsd)
    summary = {
        "aggregated_observations": len(aggregated["predictions"]),
        "aggregated_expected_aggregates": aggregated_summary["expected_aggregates"],
        "aggregated_complete_strict_aggregates": aggregated_summary["complete_strict_aggregates"],
        "aggregated_incomplete_aggregates": aggregated_summary["incomplete_aggregates"],
        "rsd_observations": len(rsd["predictions"]),
        "rsd_expected_aggregates": rsd_summary["expected_aggregates"],
        "rsd_complete_strict_aggregates": rsd_summary["complete_strict_aggregates"],
        "rsd_incomplete_aggregates": rsd_summary["incomplete_aggregates"],
        "output_root": output_root.relative_to(repository_root).as_posix(),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.repository_root.resolve())


if __name__ == "__main__":
    main()
