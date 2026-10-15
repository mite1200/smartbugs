"""Deterministic evaluation of completed reentrancy LLM experiments."""

import csv
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from experiments.reentrancy.execution import (
    build_conditions,
    build_observations,
    inspect_observation,
    load_manifest,
    load_model_configs,
    load_yaml,
)

LABELS = ("Reentrant", "Safe")
POSITIVE_LABEL = "Reentrant"
EXPECTED_REPEATS = (1, 2, 3)

PREDICTION_FIELDS = [
    "dataset",
    "dataset_name",
    "experiment_id",
    "sample_id",
    "contract_id",
    "scenario_id",
    "contract_path",
    "ground_truth",
    "model_condition",
    "model",
    "prompt_strategy",
    "repeat_index",
    "runid",
    "inference_id",
    "terminal_status",
    "parsed_status",
    "classification_valid",
    "schema_valid",
    "classification_raw",
    "prediction",
    "binary_valid",
    "technical_error",
    "source_sha256",
    "preprocessed_source_sha256",
    "analyzer_git_commit",
    "analyzer_tag",
    "provider_request_id",
    "provider_model_version",
    "api_retry_count",
    "runtime_seconds",
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "warnings",
    "result_directory",
]

AGGREGATE_FIELDS = [
    "dataset",
    "sample_id",
    "contract_id",
    "scenario_id",
    "ground_truth",
    "model_condition",
    "prompt_strategy",
    "valid_runs",
    "reentrant_votes",
    "safe_votes",
    "aggregate_status",
    "majority_prediction",
    "agreement",
    "aggregate_source",
]

METRIC_FIELDS = [
    "dataset",
    "model_condition",
    "prompt_strategy",
    "expected_samples",
    "complete_samples",
    "incomplete_samples",
    "incomplete_reentrant",
    "incomplete_safe",
    "N",
    "TP",
    "TN",
    "FP",
    "FN",
    "precision",
    "recall",
    "f1",
    "mcc",
    "accuracy",
]


def json_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def add_unique_key(keys: set[tuple[Any, ...]], key: tuple[Any, ...]) -> None:
    if key in keys:
        raise ValueError(f"duplicate observation key: {key}")
    keys.add(key)


def normalize_binary_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().casefold()
    if normalized == "reentrant":
        return "Reentrant"
    if normalized == "safe":
        return "Safe"
    return None


def binary_classification(provenance: dict[str, Any]) -> tuple[str | None, bool]:
    parsed = provenance.get("parsed_result")
    parsed_status = parsed.get("status") if isinstance(parsed, dict) else None
    classification = parsed.get("classification") if isinstance(parsed, dict) else None
    prediction = normalize_binary_label(classification)
    valid = (
        provenance.get("technical_error") is None
        and parsed_status == "completed"
        and provenance.get("classification_valid") is True
        and prediction in LABELS
    )
    return prediction, valid


def manifest_metadata(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError(f"manifest is empty: {path}")
    records: dict[str, dict[str, str]] = {}
    for row in rows:
        contract_id = row.get("contract_id", "").strip()
        scenario_id = row.get("scenario_id", "").strip()
        sample_id = scenario_id or contract_id
        contract_path = row.get("contract_path", "").strip()
        ground_truth = row.get("ground_truth", "").strip()
        if not sample_id or not contract_id or not contract_path:
            raise ValueError(f"manifest contains a malformed sample identifier: {path}")
        if ground_truth not in LABELS:
            raise ValueError(f"manifest contains unsupported ground truth {ground_truth!r}")
        if sample_id in records:
            raise ValueError(f"duplicate sample_id in manifest: {sample_id}")
        records[sample_id] = {
            "sample_id": sample_id,
            "contract_id": contract_id,
            "scenario_id": scenario_id,
            "contract_path": contract_path,
            "ground_truth": ground_truth,
        }
    return records


def _prediction_row(
    repository_root: Path,
    experiment: dict[str, Any],
    metadata: dict[str, str],
    observation: Any,
    state: Any,
) -> dict[str, Any]:
    provenance = state.provenance or {}
    parsed = provenance.get("parsed_result")
    parsed_status = parsed.get("status") if isinstance(parsed, dict) else ""
    classification_raw = parsed.get("classification") if isinstance(parsed, dict) else None
    prediction, binary_valid = binary_classification(provenance)
    usage = provenance.get("usage") or {}
    try:
        result_directory = observation.result_directory.relative_to(repository_root).as_posix()
    except ValueError:
        result_directory = observation.result_directory.as_posix()
    return {
        "dataset": experiment["dataset"],
        "dataset_name": experiment["dataset_name"],
        "experiment_id": experiment["experiment_id"],
        "sample_id": metadata["sample_id"],
        "contract_id": metadata["contract_id"],
        "scenario_id": metadata["scenario_id"],
        "contract_path": metadata["contract_path"],
        "ground_truth": metadata["ground_truth"],
        "model_condition": observation.condition.model_condition,
        "model": observation.condition.model,
        "prompt_strategy": observation.condition.strategy,
        "repeat_index": observation.condition.repeat,
        "runid": observation.condition.runid,
        "inference_id": provenance.get("inference_id", ""),
        "terminal_status": state.status,
        "parsed_status": parsed_status,
        "classification_valid": provenance.get("classification_valid", ""),
        "schema_valid": provenance.get("schema_valid", ""),
        "classification_raw": classification_raw if classification_raw is not None else "",
        "prediction": prediction or "",
        "binary_valid": binary_valid,
        "technical_error": json_cell(provenance.get("technical_error")),
        "source_sha256": provenance.get("source_sha256", ""),
        "preprocessed_source_sha256": provenance.get("preprocessed_source_sha256", ""),
        "analyzer_git_commit": provenance.get("git_commit", ""),
        "analyzer_tag": (provenance.get("metadata") or {}).get(
            "analyzer_tag", experiment["analyzer_tag"]
        ),
        "provider_request_id": provenance.get("provider_request_id", ""),
        "provider_model_version": provenance.get("provider_model_version", ""),
        "api_retry_count": provenance.get("api_retry_count", ""),
        "runtime_seconds": provenance.get("runtime_seconds", ""),
        "input_tokens": usage.get("input_tokens", ""),
        "output_tokens": usage.get("output_tokens", ""),
        "reasoning_tokens": usage.get("reasoning_tokens", ""),
        "warnings": json_cell(provenance.get("warnings", [])),
        "result_directory": result_directory,
    }


def collect_predictions(
    repository_root: Path, experiment_configs: Iterable[Path]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config_root = repository_root / "experiments/reentrancy/configs"
    models = load_model_configs(config_root / "models")
    rows: list[dict[str, Any]] = []
    observation_keys: set[tuple[Any, ...]] = set()
    inference_ids: set[str] = set()
    ground_truths: dict[tuple[str, str], str] = {}
    experiment_summaries = []

    for config_path in experiment_configs:
        experiment = load_yaml(config_path)
        repeats = tuple(sorted(int(value) for value in experiment["repeats"]))
        if repeats != EXPECTED_REPEATS:
            raise ValueError(
                f"unexpected repeat structure for {experiment['experiment_id']}: {repeats}"
            )
        results_root = repository_root / experiment["results_root"]
        if not results_root.is_dir():
            raise FileNotFoundError(f"missing expected experiment root: {results_root}")
        manifest_path = repository_root / experiment["manifest"]
        metadata_by_id = manifest_metadata(manifest_path)
        contracts = load_manifest(manifest_path)
        conditions = build_conditions(experiment, models)
        observations = build_observations(conditions, contracts, results_root)
        statuses: Counter[str] = Counter()

        for observation in observations:
            metadata = metadata_by_id.get(observation.contract.contract_id)
            if metadata is None:
                metadata = next(
                    (
                        item
                        for item in metadata_by_id.values()
                        if item["contract_id"] == observation.contract.contract_id
                    ),
                    None,
                )
            if metadata is None:
                raise ValueError(
                    f"observation is absent from manifest metadata: {observation.contract.contract_id}"
                )
            truth_key = (experiment["dataset"], metadata["sample_id"])
            existing_truth = ground_truths.setdefault(truth_key, metadata["ground_truth"])
            if existing_truth != metadata["ground_truth"]:
                raise ValueError(f"inconsistent ground truth for {truth_key}")
            state = inspect_observation(experiment, observation)
            if state.state == "conflict":
                raise RuntimeError(
                    f"conflicting raw observation {observation.result_directory}: {state.reason}"
                )
            row = _prediction_row(repository_root, experiment, metadata, observation, state)
            key = (
                row["dataset"],
                row["model_condition"],
                row["prompt_strategy"],
                row["sample_id"],
                row["repeat_index"],
            )
            add_unique_key(observation_keys, key)
            inference_id = row["inference_id"]
            if inference_id:
                if inference_id in inference_ids:
                    raise ValueError(f"duplicate inference_id: {inference_id}")
                inference_ids.add(inference_id)
            rows.append(row)
            statuses[row["terminal_status"]] += 1

        experiment_summaries.append(
            {
                "experiment_id": experiment["experiment_id"],
                "dataset": experiment["dataset"],
                "manifest": experiment["manifest"],
                "results_root": experiment["results_root"],
                "expected_observations": len(observations),
                "condition_count": len(conditions),
                "sample_count": len(contracts),
                "terminal_status_counts": dict(sorted(statuses.items())),
            }
        )

    rows.sort(
        key=lambda row: (
            row["dataset"],
            row["model_condition"],
            row["prompt_strategy"],
            row["sample_id"],
            row["repeat_index"],
        )
    )
    coverage = {
        "dataset": rows[0]["dataset"] if rows else "",
        "expected_observations": len(rows),
        "terminal_status_counts": dict(
            sorted(Counter(row["terminal_status"] for row in rows).items())
        ),
        "binary_valid_observations": sum(row["binary_valid"] is True for row in rows),
        "classification_valid_observations": sum(
            row["classification_valid"] is True for row in rows
        ),
        "schema_valid_observations": sum(row["schema_valid"] is True for row in rows),
        "experiment_ids": sorted({row["experiment_id"] for row in rows}),
        "experiments": sorted(experiment_summaries, key=lambda item: item["experiment_id"]),
        "duplicate_observation_keys": 0,
        "duplicate_inference_ids": 0,
    }
    return rows, coverage


def _group_predictions(
    rows: Iterable[dict[str, Any]],
) -> dict[tuple[str, str, str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["model_condition"],
                row["prompt_strategy"],
                row["sample_id"],
            )
        ].append(row)
    return groups


def aggregate_predictions(rows: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    if mode not in {"strict", "available"}:
        raise ValueError(f"unsupported aggregation mode: {mode}")
    aggregates = []
    for key, group in sorted(_group_predictions(rows).items()):
        repeats = tuple(sorted(int(row["repeat_index"]) for row in group))
        if repeats != EXPECTED_REPEATS:
            raise ValueError(f"unexpected repeat structure for {key}: {repeats}")
        truths = {row["ground_truth"] for row in group}
        if len(truths) != 1:
            raise ValueError(f"inconsistent ground truth for aggregate {key}")
        valid = [row for row in group if row["binary_valid"] is True]
        votes = Counter(row["prediction"] for row in valid)
        reentrant_votes = votes["Reentrant"]
        safe_votes = votes["Safe"]
        prediction = ""
        status = "incomplete"
        source = ""
        if mode == "strict" and len(valid) == 3:
            prediction = "Reentrant" if reentrant_votes >= 2 else "Safe"
            status = "complete"
            source = "3_valid_majority"
        elif mode == "available":
            if len(valid) == 3:
                prediction = "Reentrant" if reentrant_votes >= 2 else "Safe"
                status = "complete"
                source = "3_valid_majority"
            elif len(valid) == 2 and (reentrant_votes == 2 or safe_votes == 2):
                prediction = "Reentrant" if reentrant_votes == 2 else "Safe"
                status = "complete"
                source = "2_valid_agreement"
        exemplar = group[0]
        agreement = ""
        if len(valid) == 3:
            agreement = "3/3" if max(reentrant_votes, safe_votes) == 3 else "2/3"
        elif len(valid) == 2:
            agreement = "2/2" if max(reentrant_votes, safe_votes) == 2 else "1/1"
        aggregates.append(
            {
                "dataset": key[0],
                "sample_id": key[3],
                "contract_id": exemplar["contract_id"],
                "scenario_id": exemplar["scenario_id"],
                "ground_truth": next(iter(truths)),
                "model_condition": key[1],
                "prompt_strategy": key[2],
                "valid_runs": len(valid),
                "reentrant_votes": reentrant_votes,
                "safe_votes": safe_votes,
                "aggregate_status": status,
                "majority_prediction": prediction,
                "agreement": agreement,
                "aggregate_source": source,
            }
        )
    return aggregates


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def classification_metrics(
    rows: Iterable[dict[str, Any]], prediction_field: str = "majority_prediction"
) -> dict[str, int | float]:
    rows = list(rows)
    tp = sum(
        row["ground_truth"] == POSITIVE_LABEL and row[prediction_field] == POSITIVE_LABEL
        for row in rows
    )
    tn = sum(row["ground_truth"] == "Safe" and row[prediction_field] == "Safe" for row in rows)
    fp = sum(
        row["ground_truth"] == "Safe" and row[prediction_field] == POSITIVE_LABEL for row in rows
    )
    fn = sum(
        row["ground_truth"] == POSITIVE_LABEL and row[prediction_field] == "Safe" for row in rows
    )
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    accuracy = safe_divide(tp + tn, len(rows))
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = safe_divide(tp * tn - fp * fn, denominator)
    return {
        "N": len(rows),
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
        "accuracy": accuracy,
    }


def metrics_by_condition(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        groups[(row["dataset"], row["model_condition"], row["prompt_strategy"])].append(row)
    output = []
    for (dataset, model, strategy), group in sorted(groups.items()):
        complete = [row for row in group if row["aggregate_status"] == "complete"]
        incomplete = [row for row in group if row["aggregate_status"] != "complete"]
        output.append(
            {
                "dataset": dataset,
                "model_condition": model,
                "prompt_strategy": strategy,
                "expected_samples": len(group),
                "complete_samples": len(complete),
                "incomplete_samples": len(incomplete),
                "incomplete_reentrant": sum(
                    row["ground_truth"] == "Reentrant" for row in incomplete
                ),
                "incomplete_safe": sum(row["ground_truth"] == "Safe" for row in incomplete),
                **classification_metrics(complete),
            }
        )
    return output


def majority_coverage(aggregates: list[dict[str, Any]]) -> dict[str, Any]:
    rows = metrics_by_condition(aggregates)
    return {
        "expected_aggregates": len(aggregates),
        "complete_aggregates": sum(row["complete_samples"] for row in rows),
        "incomplete_aggregates": sum(row["incomplete_samples"] for row in rows),
        "by_condition": rows,
    }


def stability_rows(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        groups[(row["dataset"], row["model_condition"], row["prompt_strategy"])].append(row)
    output = []
    for (dataset, model, strategy), group in sorted(groups.items()):
        complete = [row for row in group if row["aggregate_status"] == "complete"]
        frequencies = Counter(row["reentrant_votes"] for row in complete)
        unanimous = sum(row["agreement"] == "3/3" for row in complete)
        split = sum(row["agreement"] == "2/3" for row in complete)
        output.append(
            {
                "dataset": dataset,
                "model_condition": model,
                "prompt_strategy": strategy,
                "complete_samples": len(complete),
                "unanimous_3_of_3": unanimous,
                "split_2_of_3": split,
                "unanimous_percentage": safe_divide(unanimous, len(complete)),
                "positive_votes_0_of_3": frequencies[0],
                "positive_votes_1_of_3": frequencies[1],
                "positive_votes_2_of_3": frequencies[2],
                "positive_votes_3_of_3": frequencies[3],
            }
        )
    return output


def condition_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row["dataset"],
                row["model_condition"],
                row["prompt_strategy"],
                int(row["repeat_index"]),
            )
        ].append(row)

    output = []
    aggregate_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for (dataset, model, strategy, repeat), group in sorted(groups.items()):
        aggregate_groups[(dataset, model, strategy)].extend(group)
        output.append(_condition_summary_row(dataset, model, strategy, repeat, group))
    for (dataset, model, strategy), group in sorted(aggregate_groups.items()):
        output.append(_condition_summary_row(dataset, model, strategy, "all", group))
    return sorted(
        output,
        key=lambda row: (
            row["dataset"],
            row["model_condition"],
            row["prompt_strategy"],
            str(row["repeat_index"]),
        ),
    )


def _condition_summary_row(
    dataset: str,
    model: str,
    strategy: str,
    repeat: int | str,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(row["terminal_status"] for row in rows)
    return {
        "dataset": dataset,
        "model_condition": model,
        "prompt_strategy": strategy,
        "repeat_index": repeat,
        "expected_observations": len(rows),
        "completed": statuses["completed"],
        "invalid_output": statuses["invalid_output"],
        "technical_error": statuses["technical_error"],
        "missing": statuses["missing"],
        "classification_valid": sum(row["classification_valid"] is True for row in rows),
        "schema_valid": sum(row["schema_valid"] is True for row in rows),
        "binary_valid": sum(row["binary_valid"] is True for row in rows),
    }


def available_consensus_coverage(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in aggregates:
        groups[(row["dataset"], row["model_condition"], row["prompt_strategy"])].append(row)
    output = []
    for (dataset, model, strategy), group in sorted(groups.items()):
        accepted = [row for row in group if row["aggregate_status"] == "complete"]
        output.append(
            {
                "dataset": dataset,
                "model_condition": model,
                "prompt_strategy": strategy,
                "expected_samples": len(group),
                "aggregate_samples": len(accepted),
                "coverage": safe_divide(len(accepted), len(group)),
                "from_3_valid_runs": sum(
                    row["aggregate_source"] == "3_valid_majority" for row in group
                ),
                "from_2_valid_agreement": sum(
                    row["aggregate_source"] == "2_valid_agreement" for row in group
                ),
                "rejected_samples": sum(row["aggregate_status"] != "complete" for row in group),
            }
        )
    return output


def matched_subset_metrics(
    available: list[dict[str, Any]], subset_type: str
) -> list[dict[str, Any]]:
    complete = [row for row in available if row["aggregate_status"] == "complete"]
    datasets = sorted({row["dataset"] for row in available})
    output = []
    for dataset in datasets:
        dataset_rows = [row for row in available if row["dataset"] == dataset]
        models = sorted({row["model_condition"] for row in dataset_rows})
        strategies = sorted({row["prompt_strategy"] for row in dataset_rows})
        if subset_type == "within_model":
            for model in models:
                sets = [
                    {
                        row["sample_id"]
                        for row in complete
                        if row["dataset"] == dataset
                        and row["model_condition"] == model
                        and row["prompt_strategy"] == strategy
                    }
                    for strategy in strategies
                ]
                sample_ids = set.intersection(*sets) if sets else set()
                for strategy in strategies:
                    selected = [
                        row
                        for row in complete
                        if row["dataset"] == dataset
                        and row["model_condition"] == model
                        and row["prompt_strategy"] == strategy
                        and row["sample_id"] in sample_ids
                    ]
                    output.append(
                        {
                            "dataset": dataset,
                            "subset_type": subset_type,
                            "matched_model": model,
                            "matched_strategy": "all",
                            "model_condition": model,
                            "prompt_strategy": strategy,
                            **classification_metrics(selected),
                        }
                    )
        elif subset_type == "cross_model":
            for strategy in strategies:
                sets = [
                    {
                        row["sample_id"]
                        for row in complete
                        if row["dataset"] == dataset
                        and row["model_condition"] == model
                        and row["prompt_strategy"] == strategy
                    }
                    for model in models
                ]
                sample_ids = set.intersection(*sets) if sets else set()
                for model in models:
                    selected = [
                        row
                        for row in complete
                        if row["dataset"] == dataset
                        and row["model_condition"] == model
                        and row["prompt_strategy"] == strategy
                        and row["sample_id"] in sample_ids
                    ]
                    output.append(
                        {
                            "dataset": dataset,
                            "subset_type": subset_type,
                            "matched_model": "all",
                            "matched_strategy": strategy,
                            "model_condition": model,
                            "prompt_strategy": strategy,
                            **classification_metrics(selected),
                        }
                    )
        elif subset_type == "fully_matched":
            sets = [
                {
                    row["sample_id"]
                    for row in complete
                    if row["dataset"] == dataset
                    and row["model_condition"] == model
                    and row["prompt_strategy"] == strategy
                }
                for model in models
                for strategy in strategies
            ]
            sample_ids = set.intersection(*sets) if sets else set()
            for model in models:
                for strategy in strategies:
                    selected = [
                        row
                        for row in complete
                        if row["dataset"] == dataset
                        and row["model_condition"] == model
                        and row["prompt_strategy"] == strategy
                        and row["sample_id"] in sample_ids
                    ]
                    output.append(
                        {
                            "dataset": dataset,
                            "subset_type": subset_type,
                            "matched_model": "all",
                            "matched_strategy": "all",
                            "model_condition": model,
                            "prompt_strategy": strategy,
                            **classification_metrics(selected),
                        }
                    )
        else:
            raise ValueError(f"unsupported matched subset type: {subset_type}")
    return output


def prompt_delta_rows(within_model_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in within_model_rows:
        groups[(row["dataset"], row["model_condition"])].append(row)
    output = []
    for (dataset, model), group in sorted(groups.items()):
        by_strategy = {row["prompt_strategy"]: row for row in group}
        baseline = by_strategy.get("zero_shot")
        if baseline is None:
            raise ValueError(f"zero_shot baseline missing for {(dataset, model)}")
        for strategy, row in sorted(by_strategy.items()):
            output.append(
                {
                    "dataset": dataset,
                    "model_condition": model,
                    "prompt_strategy": strategy,
                    "N": row["N"],
                    "f1": row["f1"],
                    "mcc": row["mcc"],
                    "accuracy": row["accuracy"],
                    "f1_delta_vs_zero_shot": row["f1"] - baseline["f1"],
                    "mcc_delta_vs_zero_shot": row["mcc"] - baseline["mcc"],
                    "accuracy_delta_vs_zero_shot": row["accuracy"] - baseline["accuracy"],
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not rows:
            raise ValueError(f"cannot infer CSV fields for empty output: {path}")
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_dataset(
    repository_root: Path,
    config_paths: list[Path],
    output_directory: Path,
) -> dict[str, Any]:
    predictions, coverage = collect_predictions(repository_root, config_paths)
    strict = aggregate_predictions(predictions, "strict")
    available = aggregate_predictions(predictions, "available")
    metrics = metrics_by_condition(strict)
    available_metrics = metrics_by_condition(available)
    within = matched_subset_metrics(available, "within_model")
    cross = matched_subset_metrics(available, "cross_model")
    fully = matched_subset_metrics(available, "fully_matched")

    write_csv(output_directory / "predictions.csv", predictions, PREDICTION_FIELDS)
    write_json(output_directory / "coverage.json", coverage)
    write_csv(output_directory / "majority_votes.csv", strict, AGGREGATE_FIELDS)
    write_json(output_directory / "majority_coverage.json", majority_coverage(strict))
    write_csv(output_directory / "metrics.csv", metrics, METRIC_FIELDS)
    write_csv(output_directory / "stability.csv", stability_rows(strict))
    write_csv(output_directory / "condition_summary.csv", condition_summary(predictions))
    write_csv(output_directory / "available_consensus.csv", available, AGGREGATE_FIELDS)
    write_csv(
        output_directory / "available_consensus_coverage.csv",
        available_consensus_coverage(available),
    )
    write_csv(
        output_directory / "available_consensus_metrics.csv", available_metrics, METRIC_FIELDS
    )
    write_csv(output_directory / "within_model_matched_metrics.csv", within)
    write_csv(output_directory / "cross_model_matched_metrics.csv", cross)
    write_csv(output_directory / "fully_matched_metrics.csv", fully)
    write_csv(output_directory / "prompt_deltas.csv", prompt_delta_rows(within))
    return {
        "predictions": predictions,
        "coverage": coverage,
        "strict": strict,
        "metrics": metrics,
        "available": available,
        "available_metrics": available_metrics,
        "within": within,
        "cross": cross,
        "fully": fully,
    }
