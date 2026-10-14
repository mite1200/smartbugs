import csv
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

from sb.llm.preprocessing import preprocess, remove_solidity_comments
from sb.llm.profiles import load_profile

ROOT = Path(__file__).parents[3]
BENCHMARK = Path("/tmp/reentrancy-detection-llms-rsd-prep")
SCRIPT = ROOT / "experiments/reentrancy/scripts/prepare_rsd_benchmark.py"
SPEC = importlib.util.spec_from_file_location("prepare_rsd_benchmark", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TOKEN_SCRIPT = ROOT / "experiments/reentrancy/scripts/analyze_rsd_qwen_context.py"
TOKEN_SPEC = importlib.util.spec_from_file_location("analyze_rsd_qwen_context", TOKEN_SCRIPT)
TOKEN_MODULE = importlib.util.module_from_spec(TOKEN_SPEC)
TOKEN_SPEC.loader.exec_module(TOKEN_MODULE)


def test_pinned_rsd_manifest_is_deterministic_and_complete():
    first_rows, first_metadata = MODULE.build_manifest(BENCHMARK, "fixed")
    second_rows, second_metadata = MODULE.build_manifest(BENCHMARK, "fixed")

    assert first_rows == second_rows
    assert first_metadata == second_metadata
    assert len(first_rows) == MODULE.EXPECTED_SAMPLE_COUNT == 143
    assert Counter(row["ground_truth"] for row in first_rows) == {
        "Reentrant": 71,
        "Safe": 72,
    }
    assert len({row["scenario_id"] for row in first_rows}) == len(first_rows)
    assert len({row["contract_id"] for row in first_rows}) == len(first_rows)
    assert {row["scenario_id"] for row in first_rows} == {
        path.relative_to(BENCHMARK).as_posix()
        for path in (BENCHMARK / MODULE.DATASET_ROOT).glob("*.sol")
    }


def test_pinned_rsd_paths_hashes_and_source_metadata_are_exact():
    rows, metadata = MODULE.build_manifest(BENCHMARK, "fixed")

    for row in rows:
        path = BENCHMARK / row["contract_path"]
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        assert row["contract_id"] == row["scenario_id"] == row["contract_path"]
        assert row["source"] == "RSD"
        assert row["solidity_pragma"] == "^0.8.20"
        assert row["solidity_version_family"] == "0.8"
        assert row["requires_imports"] == "false"

    assert metadata["solidity_version_distribution"] == {"0.8": 143}
    assert metadata["import_files"] == []
    assert sum(metadata["scenario_category_distribution"].values()) == 143


def test_pinned_rsd_duplicate_and_overlap_report_is_exact():
    _, metadata = MODULE.build_manifest(BENCHMARK, "fixed")

    assert metadata["unique_byte_contents"] == 142
    assert metadata["duplicate_extra_samples"] == 1
    assert metadata["cross_label_duplicate_groups"] == []
    assert metadata["duplicate_content_groups"] == [
        {
            "sha256": "69ce7e2e8693c9e43b9a45bfdeb0192422cdebeb1c47808c46dac9a92a0e1269",
            "scenario_ids": [
                "benchmarks/RSD/src/04_CrossModFold_safe1.sol",
                "benchmarks/RSD/src/04_CrossMod_safe3.sol",
            ],
            "labels": ["Safe"],
        }
    ]
    assert metadata["aggregated_exact_overlap"] == []
    assert metadata["aggregated_overlap_hash_count"] == 0
    assert metadata["aggregated_overlap_rsd_sample_count"] == 0


def test_written_rsd_csv_json_and_metadata_agree(tmp_path):
    rows, metadata = MODULE.build_manifest(BENCHMARK, "fixed")
    MODULE.write_artifacts(tmp_path, rows, metadata)

    with (tmp_path / "rsd.csv").open(encoding="utf-8", newline="") as source:
        csv_rows = list(csv.DictReader(source))
    json_rows = json.loads((tmp_path / "rsd.json").read_text(encoding="utf-8"))
    written_metadata = json.loads((tmp_path / "rsd-metadata.json").read_text(encoding="utf-8"))

    assert csv_rows == json_rows == rows
    assert written_metadata == metadata
    assert written_metadata["total_samples"] == len(rows)


def test_representative_rsd_sources_use_frozen_preprocessing_deterministically():
    profile = load_profile("reentrancy")
    source_root = BENCHMARK / MODULE.DATASET_ROOT

    for filename in (
        "00_Basic_ree1.sol",
        "09_ERC20Inherit_ree1.sol",
        "14_ReadOnly_safe1.sol",
        "15_ReadOnly_ree1.sol",
    ):
        source = (source_root / filename).read_text(encoding="utf-8")
        comment_free = remove_solidity_comments(source)
        processed = preprocess(source, profile.preprocessing)

        assert len(comment_free) == len(source)
        assert comment_free.count("\n") == source.count("\n")
        assert processed == preprocess(source, profile.preprocessing)
        assert len(processed.splitlines()) == len(source.splitlines())
        assert processed.startswith("001 | ")


def test_token_summary_distribution_reports_required_threshold_counts():
    values = [1, 2, TOKEN_MODULE.CONSERVATIVE_INPUT_LIMIT, TOKEN_MODULE.ABSOLUTE_CONTEXT_LIMIT]

    result = TOKEN_MODULE.distribution(values)

    assert result["sample_count"] == 4
    assert result["median"] == 24577.0
    assert result["p95"] == TOKEN_MODULE.ABSOLUTE_CONTEXT_LIMIT
    assert result["at_or_above_conservative_threshold"] == 2
    assert result["at_or_above_absolute_context_limit"] == 1
