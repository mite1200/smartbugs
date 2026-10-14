import importlib.util
from pathlib import Path

import csv

from sb.llm.profiles.reentrancy import STRUCTURED_REASONING

ROOT = Path(__file__).parents[3]
SCRIPT = ROOT / "experiments/reentrancy/scripts/analyze_qwen_context.py"
SPEC = importlib.util.spec_from_file_location("analyze_qwen_context", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    model_max_length = 262144

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(len(text.split())))

    def apply_chat_template(self, messages, **kwargs):
        assert messages == [{"role": "user", "content": "exact prompt"}]
        assert kwargs == {
            "tokenize": True,
            "add_generation_prompt": True,
            "reasoning_effort": "medium",
            "return_dict": True,
        }
        return {"input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}


def test_chat_token_count_uses_exact_template_parameters():
    assert MODULE.chat_input_ids(FakeTokenizer(), "exact prompt") == [1, 2, 3, 4]


def test_nearest_rank_distribution():
    values = list(range(1, 101))
    assert MODULE.distribution(values) == {
        "min": 1,
        "p50": 50,
        "p90": 90,
        "p95": 95,
        "p99": 99,
        "max": 100,
    }


def test_context_threshold_constants():
    assert MODULE.ABSOLUTE_CONTEXT_LIMIT == 65536
    assert MODULE.RESERVED_OUTPUT_TOKENS == 16384
    assert MODULE.CONSERVATIVE_INPUT_LIMIT == 49152


def test_token_count_artifact_covers_every_combined_strategy_contract():
    manifest = ROOT / "experiments/reentrancy/manifests/aggregated.csv"
    counts = (
        ROOT
        / "experiments/reentrancy/analysis/qwen3.8-27b-cerebras-context"
        / "prompt-token-counts.csv"
    )
    with manifest.open(encoding="utf-8", newline="") as source:
        contract_ids = {row["contract_id"] for row in csv.DictReader(source)}
    with counts.open(encoding="utf-8", newline="") as source:
        rows = [
            row
            for row in csv.DictReader(source)
            if row["strategy"] == "structured_reasoning_few_shot"
        ]
    keys = {(row["contract_id"], row["strategy"]) for row in rows}
    assert len(rows) == 432
    assert len(keys) == 432
    assert {row["contract_id"] for row in rows} == contract_ids
    assert all(int(row["input_tokens"]) < MODULE.ABSOLUTE_CONTEXT_LIMIT for row in rows)

    with counts.open(encoding="utf-8", newline="") as source:
        all_rows = {(row["contract_id"], row["strategy"]): row for row in csv.DictReader(source)}
    for row in rows:
        few_shot = all_rows[(row["contract_id"], "few_shot")]
        assert int(row["rendered_prompt_characters"]) == (
            int(few_shot["rendered_prompt_characters"]) + len(STRUCTURED_REASONING) + 2
        )
