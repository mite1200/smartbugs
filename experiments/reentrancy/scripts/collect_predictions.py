#!/usr/bin/env python3
"""Collect raw single-inference observations without voting or metrics."""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.reentrancy.execution import (  # noqa: E402
    build_conditions,
    build_observations,
    inspect_observation,
    load_manifest,
    load_model_configs,
    load_yaml,
)

FIELDS = [
    "contract_id",
    "contract_path",
    "ground_truth",
    "model_condition",
    "model",
    "strategy",
    "repeat",
    "runid",
    "inference_id",
    "status",
    "classification_valid",
    "schema_valid",
    "prediction",
    "technical_error",
    "api_retry_count",
    "runtime_seconds",
    "source_sha256",
    "preprocessed_source_sha256",
    "analyzer_git_commit",
    "analyzer_tag",
    "provider_request_id",
    "provider_model_version",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "warnings",
    "result_directory",
    "conflict_reason",
]


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def row_for(experiment: dict, observation: Any, state: Any) -> dict[str, Any]:
    provenance = state.provenance or {}
    usage = provenance.get("usage") or {}
    return {
        "contract_id": observation.contract.contract_id,
        "contract_path": observation.contract.contract_path,
        "ground_truth": observation.contract.ground_truth,
        "model_condition": observation.condition.model_condition,
        "model": observation.condition.model,
        "strategy": observation.condition.strategy,
        "repeat": observation.condition.repeat,
        "runid": observation.condition.runid,
        "inference_id": provenance.get("inference_id", ""),
        "status": state.status,
        "classification_valid": provenance.get("classification_valid", ""),
        "schema_valid": provenance.get("schema_valid", ""),
        "prediction": provenance.get("primary_classification", ""),
        "technical_error": json_cell(provenance.get("technical_error")),
        "api_retry_count": provenance.get("api_retry_count", ""),
        "runtime_seconds": provenance.get("runtime_seconds", ""),
        "source_sha256": provenance.get("source_sha256", ""),
        "preprocessed_source_sha256": provenance.get("preprocessed_source_sha256", ""),
        "analyzer_git_commit": provenance.get("git_commit", ""),
        "analyzer_tag": experiment["analyzer_tag"],
        "provider_request_id": provenance.get("provider_request_id", ""),
        "provider_model_version": provenance.get("provider_model_version", ""),
        "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "reasoning_tokens": usage.get("reasoning_tokens", ""),
        "warnings": json_cell(provenance.get("warnings", [])),
        "result_directory": str(observation.result_directory),
        "conflict_reason": state.reason or "",
    }


def collect(repository_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_root = repository_root / "experiments/reentrancy/configs"
    experiment = load_yaml(config_root / "experiment.yaml")
    contracts = load_manifest(repository_root / experiment["manifest"])
    conditions = build_conditions(experiment, load_model_configs(config_root / "models"))
    observations = build_observations(
        conditions,
        contracts,
        repository_root / experiment["results_root"],
    )
    rows = []
    counts: Counter[str] = Counter()
    inference_ids: Counter[str] = Counter()
    for observation in observations:
        state = inspect_observation(experiment, observation)
        rows.append(row_for(experiment, observation, state))
        counts[state.status] += 1
        if state.provenance and state.provenance.get("inference_id"):
            inference_ids[state.provenance["inference_id"]] += 1
    duplicate_ids = sorted(identifier for identifier, count in inference_ids.items() if count > 1)
    terminal_observations = (
        counts["completed"] + counts["invalid_output"] + counts["technical_error"]
    )
    summary = {
        "experiment_id": experiment["experiment_id"],
        "dataset": experiment["dataset"],
        "expected_observations": len(observations),
        "completed_observations": terminal_observations,
        "valid_completed_observations": counts["completed"],
        "missing_observations": counts["missing"],
        "invalid_output_observations": counts["invalid_output"],
        "technical_failure_observations": counts["technical_error"],
        "incomplete_or_conflict_observations": counts["incomplete"],
        "duplicate_observations": len(duplicate_ids),
        "duplicate_inference_ids": duplicate_ids,
        "condition_count": len(conditions),
        "contract_count": len(contracts),
    }
    return rows, summary


def write(repository_root: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_directory = repository_root / "experiments/reentrancy/evaluation"
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "predictions.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_directory / "coverage.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    rows, summary = collect(repository_root)
    write(repository_root, rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
