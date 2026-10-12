#!/usr/bin/env python3
"""Measure exact Qwen3.8 chat-templated prompt sizes without model inference."""

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.reentrancy.execution import load_manifest  # noqa: E402
from sb.llm.preprocessing import preprocess  # noqa: E402
from sb.llm.profiles import load_profile  # noqa: E402

MODEL_ID = "Qwen/Qwen3.8-27B"
MODEL_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
REASONING_EFFORT = "medium"
STRATEGIES = ("zero_shot", "structured_reasoning", "few_shot")
ABSOLUTE_CONTEXT_LIMIT = 65_536
RESERVED_OUTPUT_TOKENS = 16_384
CONSERVATIVE_INPUT_LIMIT = ABSOLUTE_CONTEXT_LIMIT - RESERVED_OUTPUT_TOKENS
FIELDS = [
    "contract_id",
    "strategy",
    "original_source_bytes",
    "original_source_lines",
    "preprocessed_source_bytes",
    "preprocessed_source_lines",
    "rendered_prompt_characters",
    "raw_prompt_tokens",
    "input_tokens",
    "absolute_context_ok",
    "conservative_context_ok",
]


def git_output(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def verify_benchmark(benchmark_root: Path, expected_commit: str) -> None:
    actual = git_output(benchmark_root, "rev-parse", "HEAD")
    if actual != expected_commit:
        raise RuntimeError(f"benchmark must be at {expected_commit}, found {actual}")


def load_exact_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "exact tokenization requires transformers, tokenizers, and jinja2"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
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
    """Nearest-rank percentile, returning an observed token count."""
    if not values:
        raise ValueError("cannot calculate a percentile of an empty sequence")
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value / 100 * len(ordered)))
    return ordered[rank - 1]


def distribution(values: list[int]) -> dict[str, int]:
    return {
        "min": min(values),
        "p50": percentile(values, 50),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def analyze(
    repository_root: Path,
    benchmark_root: Path,
    tokenizer: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_path = repository_root / "experiments/reentrancy/manifests/aggregated.csv"
    metadata_path = repository_root / "experiments/reentrancy/manifests/aggregated-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    verify_benchmark(benchmark_root, metadata["benchmark_commit"])
    contracts = load_manifest(manifest_path)
    profile = load_profile("reentrancy")
    rows: list[dict[str, Any]] = []
    by_strategy: dict[str, list[int]] = defaultdict(list)

    for contract in contracts:
        source_path = benchmark_root / contract.contract_path
        source_bytes = source_path.read_bytes()
        if contract.sha256 != __import__("hashlib").sha256(source_bytes).hexdigest():
            raise RuntimeError(f"source hash mismatch: {contract.contract_id}")
        source = source_bytes.decode("utf-8")
        processed = preprocess(source, profile.preprocessing)
        for strategy in STRATEGIES:
            prompt = profile.render_prompt(strategy, processed)
            raw_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            input_tokens = len(chat_input_ids(tokenizer, prompt))
            rows.append(
                {
                    "contract_id": contract.contract_id,
                    "strategy": strategy,
                    "original_source_bytes": len(source_bytes),
                    "original_source_lines": len(source.splitlines()),
                    "preprocessed_source_bytes": len(processed.encode("utf-8")),
                    "preprocessed_source_lines": len(processed.splitlines()),
                    "rendered_prompt_characters": len(prompt),
                    "raw_prompt_tokens": raw_tokens,
                    "input_tokens": input_tokens,
                    "absolute_context_ok": input_tokens < ABSOLUTE_CONTEXT_LIMIT,
                    "conservative_context_ok": input_tokens <= CONSERVATIVE_INPUT_LIMIT,
                }
            )
            by_strategy[strategy].append(input_tokens)

    rows.sort(key=lambda row: (row["contract_id"], row["strategy"]))
    largest = sorted(
        rows, key=lambda row: (-row["input_tokens"], row["contract_id"], row["strategy"])
    )[:10]
    maximum = max(row["input_tokens"] for row in rows)
    summary = {
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
        "contract_count": len(contracts),
        "strategy_count": len(STRATEGIES),
        "prompt_count": len(rows),
        "absolute_context_limit": ABSOLUTE_CONTEXT_LIMIT,
        "reserved_output_tokens": RESERVED_OUTPUT_TOKENS,
        "conservative_input_limit": CONSERVATIVE_INPUT_LIMIT,
        "strategy_distributions": {
            strategy: distribution(values) for strategy, values in sorted(by_strategy.items())
        },
        "largest_prompts": [
            {
                "contract_id": row["contract_id"],
                "strategy": row["strategy"],
                "input_tokens": row["input_tokens"],
            }
            for row in largest
        ],
        "prompts_exceeding_conservative_input_limit": sum(
            row["input_tokens"] > CONSERVATIVE_INPUT_LIMIT for row in rows
        ),
        "prompts_at_or_above_absolute_context_limit": sum(
            row["input_tokens"] >= ABSOLUTE_CONTEXT_LIMIT for row in rows
        ),
        "maximum_input_tokens": maximum,
        "worst_case_absolute_headroom": ABSOLUTE_CONTEXT_LIMIT - maximum,
        "worst_case_reserved_output_headroom": CONSERVATIVE_INPUT_LIMIT - maximum,
    }
    return rows, summary


def write(output_directory: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    with (output_directory / "prompt-token-counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output:
        writer = csv.DictWriter(output, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    (output_directory / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark_root", type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("experiments/reentrancy/analysis/qwen3.8-27b-cerebras-context"),
    )
    args = parser.parse_args()
    tokenizer = load_exact_tokenizer()
    rows, summary = analyze(
        args.repository_root.resolve(), args.benchmark_root.resolve(), tokenizer
    )
    write(args.output_directory.resolve(), rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
