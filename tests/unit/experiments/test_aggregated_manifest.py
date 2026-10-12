import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "experiments/reentrancy/scripts/generate_aggregated_manifest.py"
SPEC = importlib.util.spec_from_file_location("generate_aggregated_manifest", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def make_benchmark(tmp_path: Path) -> Path:
    root = tmp_path / "benchmark"
    (root / "benchmarks/aggregated-benchmark/src/reentrant").mkdir(parents=True)
    (root / "benchmarks/aggregated-benchmark/src/safe").mkdir(parents=True)
    (root / "benchmarks/aggregated-benchmark/src/reentrant/a_cgt.sol").write_text(
        "pragma solidity ^0.4.24;\ncontract A {}\n", encoding="utf-8"
    )
    (root / "benchmarks/aggregated-benchmark/src/safe/b.sol").write_text(
        "contract B {}\n", encoding="utf-8"
    )
    return root


def test_manifest_rows_are_deterministic_and_normalized(monkeypatch, tmp_path):
    root = make_benchmark(tmp_path)
    monkeypatch.setattr(MODULE, "verify_checkout", lambda _: None)
    first_rows, first_metadata = MODULE.build_manifest(root, "fixed")
    second_rows, second_metadata = MODULE.build_manifest(root, "fixed")
    assert first_rows == second_rows
    assert first_metadata == second_metadata
    assert [row["ground_truth"] for row in first_rows] == ["Reentrant", "Safe"]
    assert len({row["contract_id"] for row in first_rows}) == 2
    assert first_rows[0]["source"] == "cgt"
    assert first_rows[1]["source"] == "unspecified"


def test_written_csv_json_and_metadata_counts_agree(monkeypatch, tmp_path):
    root = make_benchmark(tmp_path)
    output = tmp_path / "manifests"
    monkeypatch.setattr(MODULE, "verify_checkout", lambda _: None)
    rows, metadata = MODULE.build_manifest(root, "fixed")
    MODULE.write_manifests(output, rows, metadata)
    json_rows = json.loads((output / "aggregated.json").read_text(encoding="utf-8"))
    generated_metadata = json.loads(
        (output / "aggregated-metadata.json").read_text(encoding="utf-8")
    )
    assert len(json_rows) == generated_metadata["total_contracts"] == 2
    assert generated_metadata["reentrant_count"] == 1
    assert generated_metadata["safe_count"] == 1
