#!/usr/bin/env python3
"""Prepare deterministic manifests and inspection metadata for the pinned RSD."""

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
DATASET = "Reentrancy Scenarios Dataset"
DATASET_ROOT = Path("benchmarks/RSD/src")
AGGREGATED_ROOT = Path("benchmarks/aggregated-benchmark/src")
METHOD_VERSION = "1"
EXPECTED_SAMPLE_COUNT = 143
EXPECTED_LABEL_COUNTS = {"Reentrant": 71, "Safe": 72}
FILENAME = re.compile(
    r"(?P<category>\d{2})_(?P<scenario_name>.+)_(?P<label>ree|safe)(?P<variant>\d+)\.sol"
)
PRAGMA = re.compile(r"pragma\s+solidity\s+([^;]+);", re.IGNORECASE)
VERSION = re.compile(r"(\d+)\.(\d+)(?:\.(\d+))?")
DECLARATION = re.compile(r"\b(contract|interface|library)\s+([A-Za-z_$][\w$]*)")
IMPORT = re.compile(r"^\s*import\b", re.MULTILINE)
FIELDNAMES = [
    "contract_id",
    "scenario_id",
    "contract_path",
    "ground_truth",
    "dataset",
    "source",
    "sha256",
    "solidity_pragma",
    "solidity_version_family",
    "scenario_category",
    "scenario_name",
    "scenario_variant",
    "declaration_count",
    "requires_imports",
]


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_checkout(benchmark_root: Path) -> None:
    actual = git_output(benchmark_root, "rev-parse", "HEAD")
    if actual != BENCHMARK_COMMIT:
        raise ValueError(f"benchmark checkout must be at commit {BENCHMARK_COMMIT}, found {actual}")
    status = git_output(benchmark_root, "status", "--porcelain", "--", DATASET_ROOT.as_posix())
    if status:
        raise ValueError("RSD source tree contains uncommitted changes")


def pragma_metadata(source: bytes) -> tuple[str, str, list[str]]:
    text = remove_comments(source.decode("utf-8"))
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


def remove_comments(source: str) -> str:
    """Remove Solidity comments while preserving character and newline positions."""
    result = list(source)
    state = "code"
    quote = ""
    index = 0
    while index < len(source):
        char = source[index]
        following = source[index + 1] if index + 1 < len(source) else ""
        if state == "code":
            if char in {'"', "'"}:
                state, quote = "string", char
            elif char == "/" and following == "/":
                result[index] = result[index + 1] = " "
                state = "line_comment"
                index += 1
            elif char == "/" and following == "*":
                result[index] = result[index + 1] = " "
                state = "block_comment"
                index += 1
        elif state == "string":
            if char == "\\":
                state = "escape"
            elif char == quote:
                state = "code"
        elif state == "escape":
            state = "string"
        elif state == "line_comment":
            if char in "\r\n":
                state = "code"
            else:
                result[index] = " "
        elif state == "block_comment":
            if char == "*" and following == "/":
                result[index] = result[index + 1] = " "
                state = "code"
                index += 1
            elif char not in "\r\n":
                result[index] = " "
        index += 1
    return "".join(result)


def label_for_filename(path: Path) -> tuple[re.Match[str], str]:
    match = FILENAME.fullmatch(path.name)
    if not match:
        raise ValueError(f"RSD filename does not encode the documented scenario label: {path.name}")
    return match, "Reentrant" if match.group("label") == "ree" else "Safe"


def source_groups(paths: list[Path], root: Path) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups[digest].append(path.relative_to(root).as_posix())
    return groups


def label_from_aggregated_path(path: Path) -> str:
    if path.parent.name == "reentrant":
        return "Reentrant"
    if path.parent.name == "safe":
        return "Safe"
    raise ValueError(f"unexpected Aggregated label directory: {path}")


def build_manifest(
    benchmark_root: Path, timestamp: str
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    verify_checkout(benchmark_root)
    source_root = benchmark_root / DATASET_ROOT
    if not source_root.is_dir():
        raise ValueError(f"missing RSD source directory: {source_root}")

    source_paths = sorted(path for path in source_root.rglob("*.sol") if path.is_file())
    rows: list[dict[str, str]] = []
    label_distribution: Counter[str] = Counter()
    version_distribution: Counter[str] = Counter()
    category_distribution: Counter[str] = Counter()
    declaration_distribution: Counter[str] = Counter()
    content_groups: dict[str, list[str]] = defaultdict(list)
    multiple_pragma_files: list[dict[str, Any]] = []
    multi_declaration_files: list[dict[str, Any]] = []
    import_files: list[str] = []

    for path in source_paths:
        match, ground_truth = label_for_filename(path)
        source_bytes = path.read_bytes()
        text = source_bytes.decode("utf-8")
        active_text = remove_comments(text)
        relative_path = path.relative_to(benchmark_root).as_posix()
        digest = hashlib.sha256(source_bytes).hexdigest()
        pragma, version_family, all_pragmas = pragma_metadata(source_bytes)
        declarations = DECLARATION.findall(active_text)
        requires_imports = bool(IMPORT.search(active_text))
        row = {
            "contract_id": relative_path,
            "scenario_id": relative_path,
            "contract_path": relative_path,
            "ground_truth": ground_truth,
            "dataset": DATASET,
            "source": "RSD",
            "sha256": digest,
            "solidity_pragma": pragma,
            "solidity_version_family": version_family,
            "scenario_category": match.group("category"),
            "scenario_name": match.group("scenario_name"),
            "scenario_variant": match.group("variant"),
            "declaration_count": str(len(declarations)),
            "requires_imports": str(requires_imports).lower(),
        }
        rows.append(row)
        label_distribution[ground_truth] += 1
        version_distribution[version_family] += 1
        category_distribution[match.group("category")] += 1
        declaration_distribution[str(len(declarations))] += 1
        content_groups[digest].append(relative_path)
        if len(all_pragmas) > 1:
            multiple_pragma_files.append({"scenario_id": relative_path, "pragmas": all_pragmas})
        if len(declarations) > 1:
            multi_declaration_files.append(
                {
                    "scenario_id": relative_path,
                    "declarations": [{"kind": kind, "name": name} for kind, name in declarations],
                }
            )
        if requires_imports:
            import_files.append(relative_path)

    rows.sort(key=lambda row: row["scenario_id"])
    duplicate_content = [
        {
            "sha256": digest,
            "scenario_ids": paths,
            "labels": sorted({row["ground_truth"] for row in rows if row["scenario_id"] in paths}),
        }
        for digest, paths in sorted(content_groups.items())
        if len(paths) > 1
    ]

    aggregated_paths = sorted(
        path for path in (benchmark_root / AGGREGATED_ROOT).rglob("*.sol") if path.is_file()
    )
    aggregated_groups = source_groups(aggregated_paths, benchmark_root)
    overlap = []
    row_by_id = {row["scenario_id"]: row for row in rows}
    for digest in sorted(set(content_groups) & set(aggregated_groups)):
        rsd_ids = content_groups[digest]
        aggregated_ids = aggregated_groups[digest]
        rsd_labels = sorted({row_by_id[scenario_id]["ground_truth"] for scenario_id in rsd_ids})
        aggregated_labels = sorted(
            {
                label_from_aggregated_path(benchmark_root / contract_path)
                for contract_path in aggregated_ids
            }
        )
        overlap.append(
            {
                "sha256": digest,
                "rsd_scenario_ids": rsd_ids,
                "rsd_labels": rsd_labels,
                "aggregated_contract_ids": aggregated_ids,
                "aggregated_labels": aggregated_labels,
                "labels_agree": rsd_labels == aggregated_labels,
            }
        )

    metadata: dict[str, Any] = {
        "benchmark_repository": BENCHMARK_REPOSITORY,
        "benchmark_commit": BENCHMARK_COMMIT,
        "generation_timestamp": timestamp,
        "dataset": DATASET,
        "dataset_root": DATASET_ROOT.as_posix(),
        "total_samples": len(rows),
        "reentrant_count": label_distribution["Reentrant"],
        "safe_count": label_distribution["Safe"],
        "other_label_count": len(rows)
        - label_distribution["Reentrant"]
        - label_distribution["Safe"],
        "class_distribution": dict(sorted(label_distribution.items())),
        "solidity_version_distribution": dict(sorted(version_distribution.items())),
        "scenario_category_distribution": dict(sorted(category_distribution.items())),
        "declaration_count_distribution": dict(sorted(declaration_distribution.items())),
        "multi_declaration_files": multi_declaration_files,
        "import_files": import_files,
        "multiple_pragma_files": multiple_pragma_files,
        "unique_byte_contents": len(content_groups),
        "duplicate_content_groups": duplicate_content,
        "duplicate_extra_samples": sum(
            len(group["scenario_ids"]) - 1 for group in duplicate_content
        ),
        "cross_label_duplicate_groups": [
            group for group in duplicate_content if len(group["labels"]) > 1
        ],
        "aggregated_exact_overlap": overlap,
        "aggregated_overlap_hash_count": len(overlap),
        "aggregated_overlap_rsd_sample_count": sum(
            len(group["rsd_scenario_ids"]) for group in overlap
        ),
        "sample_unit": (
            "One Solidity source file in benchmarks/RSD/src; each file is one labeled scenario "
            "and may contain multiple contract/interface declarations"
        ),
        "scenario_id_rule": (
            "POSIX-normalized path relative to the benchmark repository root; "
            "scenario_id, contract_id, and contract_path are equal"
        ),
        "ground_truth_rule": (
            "Terminal filename component _reeN -> Reentrant and _safeN -> Safe; this is the "
            "label encoding used by the pinned dataset and its per-file result tables"
        ),
        "scenario_metadata_rule": (
            "scenario_category is the leading two-digit filename component; scenario_name and "
            "scenario_variant preserve the remaining filename components without semantic inference"
        ),
        "solidity_version_rule": (
            "Major.minor family extracted from all Solidity pragma directives; all directives "
            "must agree, otherwise 'multiple'; '(no pragma)' when absent"
        ),
        "manifest_generation_method_version": METHOD_VERSION,
        "documented_count_discrepancy": {
            "readme_counts_seen": [150, 154],
            "pinned_source_count": len(rows),
            "stale_result_filename_count": 154,
            "removed_source_count": 11,
            "renamed_source_count": 2,
        },
    }
    return rows, metadata


def validate_authoritative_counts(rows: list[dict[str, str]], metadata: dict[str, Any]) -> None:
    labels = Counter(row["ground_truth"] for row in rows)
    if len(rows) != EXPECTED_SAMPLE_COUNT:
        raise ValueError(f"expected {EXPECTED_SAMPLE_COUNT} pinned RSD samples, found {len(rows)}")
    if dict(labels) != EXPECTED_LABEL_COUNTS:
        raise ValueError(f"unexpected pinned RSD class distribution: {dict(labels)}")
    if metadata["total_samples"] != len(rows):
        raise ValueError("metadata total_samples does not match manifest")


def write_artifacts(
    output_directory: Path, rows: list[dict[str, str]], metadata: dict[str, Any]
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "rsd.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_directory / "rsd.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_directory / "rsd-metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/reentrancy/benchmarks/rsd"),
    )
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    timestamp = args.timestamp or datetime.now(timezone.utc).isoformat()
    rows, metadata = build_manifest(args.benchmark_root.resolve(), timestamp)
    validate_authoritative_counts(rows, metadata)
    write_artifacts(args.output_directory.resolve(), rows, metadata)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
