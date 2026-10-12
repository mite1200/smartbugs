#!/usr/bin/env python3
"""Generate canonical manifests for the pinned Aggregated Benchmark."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_REPOSITORY = (
    "https://github.com/ca-foscari-reentrancy-research-group/" "reentrancy-detection-llms.git"
)
BENCHMARK_COMMIT = "d7c9496121931cff7c55b2047d18fb048fefe2cb"
DATASET = "Aggregated Benchmark"
DATASET_ROOT = Path("benchmarks/aggregated-benchmark/src")
METHOD_VERSION = "1"
FIELDNAMES = [
    "contract_id",
    "contract_path",
    "ground_truth",
    "dataset",
    "source",
    "sha256",
    "solidity_pragma",
    "solidity_version_family",
]
LABELS = {"reentrant": "Reentrant", "safe": "Safe"}
SOURCE_SUFFIX = re.compile(r"_(cgt|hg|rs)$", re.IGNORECASE)
PRAGMA = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)
VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")


def git_output(root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def verify_checkout(benchmark_root: Path) -> None:
    if git_output(benchmark_root, "rev-parse", "HEAD") != BENCHMARK_COMMIT:
        raise ValueError(f"benchmark checkout must be at commit {BENCHMARK_COMMIT}")
    status = git_output(benchmark_root, "status", "--porcelain", "--", DATASET_ROOT.as_posix())
    if status:
        raise ValueError("benchmark source tree contains uncommitted changes")


def source_name(path: Path) -> str:
    match = SOURCE_SUFFIX.search(path.stem)
    return match.group(1).lower() if match else "unspecified"


def pragma_metadata(source: bytes) -> tuple[str, str, list[str]]:
    text = source.decode("utf-8")
    pragmas = [value.strip() for value in PRAGMA.findall(text)]
    if not pragmas:
        return "", "(no pragma)", []
    families = []
    for pragma in pragmas:
        match = VERSION.search(pragma)
        families.append(f"{match.group(1)}.{match.group(2)}" if match else "(unparsed pragma)")
    unique_families = sorted(set(families))
    family = unique_families[0] if len(unique_families) == 1 else "multiple"
    return pragmas[0], family, pragmas


def build_manifest(
    benchmark_root: Path, timestamp: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    verify_checkout(benchmark_root)
    source_root = benchmark_root / DATASET_ROOT
    rows: list[dict[str, str]] = []
    content_groups: dict[str, list[str]] = defaultdict(list)
    basename_groups: dict[str, list[str]] = defaultdict(list)
    version_distribution: Counter[str] = Counter()
    source_distribution: Counter[str] = Counter()
    label_distribution: Counter[str] = Counter()
    multiple_pragma_files: list[dict[str, Any]] = []

    for label_directory, ground_truth in LABELS.items():
        directory = source_root / label_directory
        if not directory.is_dir():
            raise ValueError(f"missing label directory: {directory}")
        for path in sorted(directory.rglob("*.sol")):
            if not path.is_file():
                continue
            source_bytes = path.read_bytes()
            relative_path = path.relative_to(benchmark_root).as_posix()
            digest = hashlib.sha256(source_bytes).hexdigest()
            pragma, version_family, all_pragmas = pragma_metadata(source_bytes)
            source = source_name(path)
            row = {
                "contract_id": relative_path,
                "contract_path": relative_path,
                "ground_truth": ground_truth,
                "dataset": DATASET,
                "source": source,
                "sha256": digest,
                "solidity_pragma": pragma,
                "solidity_version_family": version_family,
            }
            rows.append(row)
            content_groups[digest].append(relative_path)
            basename_groups[path.name].append(relative_path)
            version_distribution[version_family] += 1
            source_distribution[source] += 1
            label_distribution[ground_truth] += 1
            if len(all_pragmas) > 1:
                multiple_pragma_files.append({"contract_id": relative_path, "pragmas": all_pragmas})

    rows.sort(key=lambda row: row["contract_id"])
    duplicate_content = [
        {"sha256": digest, "contract_ids": paths}
        for digest, paths in sorted(content_groups.items())
        if len(paths) > 1
    ]
    duplicate_filenames = [
        {"filename": filename, "contract_ids": paths}
        for filename, paths in sorted(basename_groups.items())
        if len(paths) > 1
    ]
    metadata: dict[str, Any] = {
        "benchmark_repository": BENCHMARK_REPOSITORY,
        "benchmark_commit": BENCHMARK_COMMIT,
        "generation_timestamp": timestamp,
        "dataset": DATASET,
        "dataset_root": DATASET_ROOT.as_posix(),
        "total_contracts": len(rows),
        "reentrant_count": label_distribution["Reentrant"],
        "safe_count": label_distribution["Safe"],
        "source_distribution": dict(sorted(source_distribution.items())),
        "duplicate_content_groups": duplicate_content,
        "duplicate_filename_groups": duplicate_filenames,
        "solidity_version_distribution": dict(sorted(version_distribution.items())),
        "multiple_pragma_files": multiple_pragma_files,
        "unreadable_files": [],
        "contract_id_rule": (
            "POSIX-normalized path relative to the benchmark repository root; "
            "contract_id equals contract_path"
        ),
        "ground_truth_rule": (
            "Immediate dataset directory: src/reentrant -> Reentrant, src/safe -> Safe"
        ),
        "source_rule": ("Terminal filename suffix _cgt, _hg, or _rs; otherwise unspecified"),
        "solidity_version_rule": (
            "Major.minor family extracted from all Solidity pragma directives; "
            "all directives must agree, otherwise 'multiple'; '(no pragma)' when absent"
        ),
        "manifest_generation_method_version": METHOD_VERSION,
        "readme_count_discrepancy": {
            "readme_total": 436,
            "readme_reentrant": 122,
            "readme_safe": 314,
            "derived_total": len(rows),
            "derived_reentrant": label_distribution["Reentrant"],
            "derived_safe": label_distribution["Safe"],
        },
    }
    return rows, metadata


def write_manifests(
    output_directory: Path, rows: list[dict[str, str]], metadata: dict[str, Any]
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "aggregated.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_directory / "aggregated.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "aggregated-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/reentrancy/manifests"),
    )
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    rows, metadata = build_manifest(args.benchmark_root.resolve(), timestamp)
    write_manifests(args.output_directory.resolve(), rows, metadata)


if __name__ == "__main__":
    main()
