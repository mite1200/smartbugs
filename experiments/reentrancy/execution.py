"""Deterministic experiment planning and conservative resume classification."""

import csv
import hashlib
import json
import os
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

TERMINAL_STATUSES = {"completed", "invalid_output", "technical_error"}
STRATEGY_SLUGS = {
    "zero_shot": "zero-shot",
    "structured_reasoning": "structured-reasoning",
    "few_shot": "few-shot",
}
PROTECTED_PATHS = (
    "sb/llm",
    "tools/llm",
    "sb/analysis.py",
    "sb/docker.py",
    "sb/settings.py",
    "sb/smartbugs.py",
    "sb/tools.py",
    "sb/runners.py",
)


@dataclass(frozen=True)
class Contract:
    contract_id: str
    contract_path: str
    ground_truth: str
    sha256: str


@dataclass(frozen=True)
class Condition:
    dataset: str
    model_condition: str
    model: str
    provider: dict[str, Any]
    strategy: str
    repeat: int
    runid: str


@dataclass(frozen=True)
class Observation:
    condition: Condition
    contract: Contract
    result_directory: Path


@dataclass(frozen=True)
class ObservationState:
    state: str
    status: str
    provenance: dict[str, Any] | None = None
    reason: str | None = None


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(value, dict):
        raise ValueError(f"mapping expected in {path}")
    return value


def load_manifest(path: Path) -> list[Contract]:
    with path.open(encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    contracts = [
        Contract(
            contract_id=row["contract_id"],
            contract_path=row["contract_path"],
            ground_truth=row["ground_truth"],
            sha256=row["sha256"],
        )
        for row in rows
    ]
    if len({contract.contract_id for contract in contracts}) != len(contracts):
        raise ValueError("manifest contract_id values are not unique")
    return sorted(contracts, key=lambda contract: contract.contract_id)


def load_model_configs(directory: Path) -> list[dict[str, Any]]:
    configs = [load_yaml(path) for path in sorted(directory.glob("*.yaml"))]
    identifiers = [config.get("condition_id") for config in configs]
    if not configs or any(not isinstance(identifier, str) for identifier in identifiers):
        raise ValueError("every model configuration needs a string condition_id")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("model condition identifiers must be unique")
    return configs


def make_runid(dataset: str, model_condition: str, strategy: str, repeat: int) -> str:
    return f"{dataset}__{model_condition}__{STRATEGY_SLUGS[strategy]}__r{repeat}"


def build_conditions(experiment: dict[str, Any], models: list[dict[str, Any]]) -> list[Condition]:
    conditions = []
    for model_config in models:
        provider = model_config["provider"]
        for strategy in experiment["strategies"]:
            if strategy not in STRATEGY_SLUGS:
                raise ValueError(f"unsupported experiment strategy: {strategy}")
            for repeat in experiment["repeats"]:
                conditions.append(
                    Condition(
                        dataset=experiment["dataset"],
                        model_condition=model_config["condition_id"],
                        model=provider["model"],
                        provider=provider,
                        strategy=strategy,
                        repeat=int(repeat),
                        runid=make_runid(
                            experiment["dataset"],
                            model_config["condition_id"],
                            strategy,
                            int(repeat),
                        ),
                    )
                )
    return sorted(conditions, key=lambda condition: condition.runid)


def result_directory(results_root: Path, condition: Condition, contract: Contract) -> Path:
    return results_root / condition.runid / Path(contract.contract_path)


def build_observations(
    conditions: Iterable[Condition], contracts: Iterable[Contract], results_root: Path
) -> list[Observation]:
    return [
        Observation(condition, contract, result_directory(results_root, condition, contract))
        for condition in conditions
        for contract in contracts
    ]


def expected_metadata(experiment: dict[str, Any], observation: Observation) -> dict[str, Any]:
    return {
        "experiment_id": experiment["experiment_id"],
        "dataset": observation.condition.dataset,
        "benchmark_commit": experiment["benchmark_commit"],
        "analyzer_tag": experiment["analyzer_tag"],
        "repeat_index": observation.condition.repeat,
        "model_condition": observation.condition.model_condition,
        "prompt_strategy": observation.condition.strategy,
    }


def provenance_status(provenance: dict[str, Any]) -> str | None:
    if provenance.get("technical_error") is not None:
        return "technical_error"
    parsed = provenance.get("parsed_result")
    if isinstance(parsed, dict) and parsed.get("status") in {"completed", "invalid_output"}:
        return parsed["status"]
    return None


def inspect_observation(experiment: dict[str, Any], observation: Observation) -> ObservationState:
    directory = observation.result_directory
    provenance_path = directory / "llm-provenance.json"
    if provenance_path.is_file():
        try:
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            return ObservationState("conflict", "incomplete", reason=f"invalid provenance: {error}")
        checks = {
            "contract_path": observation.contract.contract_path,
            "source_sha256": observation.contract.sha256,
            "model": observation.condition.model,
            "prompt_strategy": observation.condition.strategy,
        }
        for key, expected in checks.items():
            if provenance.get(key) != expected:
                return ObservationState(
                    "conflict",
                    "incomplete",
                    provenance,
                    f"provenance {key} mismatch",
                )
        metadata = provenance.get("metadata")
        if not isinstance(metadata, dict):
            return ObservationState("conflict", "incomplete", provenance, "metadata missing")
        for key, expected in expected_metadata(experiment, observation).items():
            if metadata.get(key) != expected:
                return ObservationState(
                    "conflict",
                    "incomplete",
                    provenance,
                    f"metadata {key} mismatch",
                )
        status = provenance_status(provenance)
        if status not in TERMINAL_STATUSES:
            return ObservationState("conflict", "incomplete", provenance, "unknown status")
        return ObservationState("completed", status, provenance)
    if directory.exists():
        try:
            entries = list(directory.iterdir())
        except OSError as error:
            return ObservationState("conflict", "incomplete", reason=str(error))
        if entries:
            return ObservationState(
                "conflict",
                "incomplete",
                reason="result artifacts exist without matching provenance",
            )
        return ObservationState("conflict", "incomplete", reason="empty result directory exists")
    return ObservationState("missing", "missing")


def manifest_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_benchmark(
    benchmark_root: Path, experiment: dict[str, Any], contracts: Iterable[Contract]
) -> None:
    commit = git_output(benchmark_root, "rev-parse", "HEAD")
    if commit != experiment["benchmark_commit"]:
        raise RuntimeError(
            f"benchmark checkout must be at {experiment['benchmark_commit']}, found {commit}"
        )
    for contract in contracts:
        path = benchmark_root / contract.contract_path
        if not path.is_file():
            raise FileNotFoundError(f"benchmark source missing: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != contract.sha256:
            raise RuntimeError(f"benchmark source hash mismatch: {contract.contract_id}")


def verify_frozen_analyzer(repository_root: Path, tag: str) -> str:
    tag_commit = git_output(repository_root, "rev-list", "-n", "1", tag)
    changed = git_output(repository_root, "diff", "--name-only", tag, "--", *PROTECTED_PATHS)
    if changed:
        raise RuntimeError(f"frozen analyzer/core differs from {tag}: {changed}")
    status = git_output(repository_root, "status", "--porcelain", "--", *PROTECTED_PATHS)
    if status:
        raise RuntimeError(f"frozen analyzer/core has local changes: {status}")
    return tag_commit


class ConditionLock:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.fd: int | None = None

    def __enter__(self) -> "ConditionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as error:
            raise RuntimeError(f"condition is already locked: {self.path}") from error
        os.write(self.fd, f"pid={os.getpid()}\n".encode())
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.fd is not None:
            os.close(self.fd)
        self.path.unlink(missing_ok=True)
