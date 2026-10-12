#!/usr/bin/env python3
"""Validate generated Aggregated Benchmark manifests against source bytes."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

ALLOWED_LABELS = {"Reentrant", "Safe"}


def load_files(
    manifest_directory: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    with (manifest_directory / "aggregated.csv").open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    json_rows = json.loads((manifest_directory / "aggregated.json").read_text(encoding="utf-8"))
    metadata = json.loads(
        (manifest_directory / "aggregated-metadata.json").read_text(encoding="utf-8")
    )
    return csv_rows, json_rows, metadata


def validate(benchmark_root: Path, manifest_directory: Path) -> None:
    csv_rows, json_rows, metadata = load_files(manifest_directory)
    if csv_rows != json_rows:
        raise AssertionError("CSV and JSON manifest rows differ")
    ids = [row["contract_id"] for row in json_rows]
    if len(ids) != len(set(ids)):
        raise AssertionError("contract_id values are not unique")
    labels = Counter(row["ground_truth"] for row in json_rows)
    if not set(labels).issubset(ALLOWED_LABELS):
        raise AssertionError("manifest contains an unsupported ground_truth label")
    for row in json_rows:
        path = benchmark_root / row["contract_path"]
        if not path.is_file():
            raise AssertionError(f"referenced contract does not exist: {path}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise AssertionError(f"SHA-256 mismatch for {row['contract_id']}")
        if row["contract_id"] != row["contract_path"]:
            raise AssertionError("contract_id does not follow the documented path rule")
    if metadata["total_contracts"] != len(json_rows):
        raise AssertionError("metadata total_contracts does not match manifests")
    if metadata["reentrant_count"] != labels["Reentrant"]:
        raise AssertionError("metadata reentrant_count does not match manifests")
    if metadata["safe_count"] != labels["Safe"]:
        raise AssertionError("metadata safe_count does not match manifests")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument(
        "--manifest-directory",
        type=Path,
        default=Path("experiments/reentrancy/manifests"),
    )
    args = parser.parse_args()
    validate(args.benchmark_root.resolve(), args.manifest_directory.resolve())


if __name__ == "__main__":
    main()
