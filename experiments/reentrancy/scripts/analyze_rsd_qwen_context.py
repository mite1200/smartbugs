#!/usr/bin/env python3
"""Measure exact frozen Qwen prompt sizes for the prepared RSD manifest."""

import argparse
import csv
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from sb.llm.preprocessing import preprocess  # noqa: E402
from sb.llm.profiles import load_profile  # noqa: E402

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REASONING_EFFORT = "medium"
BENCHMARK_COMMIT = "d7c9496121931cff7c55b2047d18fb048fefe2cb"
STRATEGIES = (
    "zero_shot",
    "structured_reasoning",
    "few_shot",
    "structured_reasoning_few_shot",
)
ABSOLUTE_CONTEXT_LIMIT = 65_536
CONSERVATIVE_INPUT_LIMIT = 49_152
FIELDS = [
    "scenario_id",
    "strategy",
    "original_source_bytes",
    "original_source_lines",
    "preprocessed_source_bytes",
    "preprocessed_source_lines",
    "rendered_prompt_characters",
    "input_tokens",
    "at_or_above_conservative_threshold",
    "at_or_above_absolute_context_limit",
]


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def verify_benchmark(benchmark_root: Path) -> None:
    actual = git_output(benchmark_root, "rev-parse", "HEAD")
    if actual != BENCHMARK_COMMIT:
        raise RuntimeError(f"benchmark must be at {BENCHMARK_COMMIT}, found {actual}")


def load_tokenizer(tokenizer_path: str | None) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "exact tokenization requires transformers, tokenizers, and jinja2"
        ) from error
    source = tokenizer_path or MODEL_ID
    kwargs: dict[str, Any] = {"local_files_only": bool(tokenizer_path)}
    if not tokenizer_path:
        kwargs["revision"] = MODEL_REVISION
    tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
    if not tokenizer.chat_template:
        raise RuntimeError("pinned Qwen tokenizer has no chat template")
    return tokenizer


def chat_input_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        reasoning_effort=REASONING_EFFORT,
        return_dict=True,
    )
    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], list):
        if len(input_ids) != 1:
            raise RuntimeError("unexpected batched chat-template encoding")
        input_ids = input_ids[0]
    return input_ids


def percentile(values: list[int], percentile_value: float) -> int:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value / 100 * len(ordered)))
    return ordered[rank - 1]


def distribution(values: list[int]) -> dict[str, int | float]:
    return {
        "sample_count": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "p95": percentile(values, 95),
        "max": max(values),
        "total_input_tokens": sum(values),
        "at_or_above_conservative_threshold": sum(
            value >= CONSERVATIVE_INPUT_LIMIT for value in values
        ),
        "at_or_above_absolute_context_limit": sum(
            value >= ABSOLUTE_CONTEXT_LIMIT for value in values
        ),
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as source:
        return list(csv.DictReader(source))


def analyze(
    benchmark_root: Path, manifest_path: Path, tokenizer: Any
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verify_benchmark(benchmark_root)
    manifest = load_manifest(manifest_path)
    profile = load_profile("reentrancy")
    rows: list[dict[str, Any]] = []
    by_strategy: dict[str, list[int]] = defaultdict(list)

    for sample in manifest:
        source_path = benchmark_root / sample["contract_path"]
        source_bytes = source_path.read_bytes()
        if hashlib.sha256(source_bytes).hexdigest() != sample["sha256"]:
            raise RuntimeError(f"source hash mismatch: {sample['scenario_id']}")
        source = source_bytes.decode("utf-8")
        processed = preprocess(source, profile.preprocessing)
        for strategy in STRATEGIES:
            prompt = profile.render_prompt(strategy, processed)
            input_tokens = len(chat_input_ids(tokenizer, prompt))
            rows.append(
                {
                    "scenario_id": sample["scenario_id"],
                    "strategy": strategy,
                    "original_source_bytes": len(source_bytes),
                    "original_source_lines": len(source.splitlines()),
                    "preprocessed_source_bytes": len(processed.encode("utf-8")),
                    "preprocessed_source_lines": len(processed.splitlines()),
                    "rendered_prompt_characters": len(prompt),
                    "input_tokens": input_tokens,
                    "at_or_above_conservative_threshold": input_tokens >= CONSERVATIVE_INPUT_LIMIT,
                    "at_or_above_absolute_context_limit": input_tokens >= ABSOLUTE_CONTEXT_LIMIT,
                }
            )
            by_strategy[strategy].append(input_tokens)

    rows.sort(key=lambda row: (row["scenario_id"], row["strategy"]))
    summary = {
        "benchmark_commit": BENCHMARK_COMMIT,
        "manifest": manifest_path.as_posix(),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_model_max_length": tokenizer.model_max_length,
        "count_type": "chat-templated input tokens",
        "chat_template_parameters": {
            "messages": "one user message containing the exact rendered analyzer prompt",
            "add_generation_prompt": True,
            "reasoning_effort": REASONING_EFFORT,
        },
        "sample_count": len(manifest),
        "strategy_count": len(STRATEGIES),
        "prompt_count": len(rows),
        "conservative_input_threshold": CONSERVATIVE_INPUT_LIMIT,
        "absolute_context_limit": ABSOLUTE_CONTEXT_LIMIT,
        "strategy_distributions": {
            strategy: distribution(values) for strategy, values in sorted(by_strategy.items())
        },
    }
    return rows, summary


def write(output_directory: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "qwen-prompt-token-counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_directory / "qwen-token-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/reentrancy/benchmarks/rsd/rsd.csv"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/reentrancy/benchmarks/rsd"),
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Optional local snapshot of the exact pinned tokenizer; avoids network access",
    )
    args = parser.parse_args()
    tokenizer = load_tokenizer(args.tokenizer_path)
    rows, summary = analyze(args.benchmark_root.resolve(), args.manifest.resolve(), tokenizer)
    write(args.output_directory.resolve(), rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
