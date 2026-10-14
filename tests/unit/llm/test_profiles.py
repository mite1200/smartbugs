import hashlib

from sb.llm.preprocessing import preprocess
from sb.llm.profiles import load_profile
from sb.llm.profiles.reentrancy import (
    COMMON_OUTPUT,
    DEMONSTRATIONS,
    REENTRANCY_DEFINITION,
    STRUCTURED_REASONING,
)

EXISTING_PROMPT_HASHES = {
    "zero_shot": "39539f4209ba71b25ac5c145cc1c6557de3f8fcf47f2b0d9e544324551fc0105",
    "structured_reasoning": "83a721faa9b75ffe4cd3fd9c009273138e431585c8ca2ed1e4cf506b7904255a",
    "few_shot": "2762bf8876ad7022eff4b6cca9c078c626ae9283f695cac1669525ad27e3465d",
}


def test_reentrancy_profile_loads_and_is_versioned():
    profile = load_profile("reentrancy")
    assert profile.id == "reentrancy"
    assert profile.version == 1
    assert profile.prompt_versions["zero_shot"] == 1
    assert profile.preprocessing.remove_comments is True
    assert profile.preprocessing.add_line_numbers is True


def test_strategy_interventions_are_isolated():
    profile = load_profile("reentrancy")
    source = "001 | contract Target {}"
    zero = profile.render_prompt("zero_shot", source)
    structured = profile.render_prompt("structured_reasoning", source)
    few = profile.render_prompt("few_shot", source)
    for prompt in (zero, structured, few):
        assert prompt.count(REENTRANCY_DEFINITION) == 1
        assert source in prompt
    assert STRUCTURED_REASONING not in zero
    assert "Example 1" not in zero
    assert STRUCTURED_REASONING in structured
    assert "Example 1" not in structured
    assert STRUCTURED_REASONING not in few
    assert few.count("Expected output:") == 4
    assert (
        few.index("Example 1")
        < few.index("Example 2")
        < few.index("Example 3")
        < few.index("Example 4")
    )
    assert DEMONSTRATIONS in few


def test_combined_strategy_composes_frozen_prompt_components():
    profile = load_profile("reentrancy")
    source = "001 | contract Target {}"
    combined = profile.render_prompt("structured_reasoning_few_shot", source)
    assert combined.count(REENTRANCY_DEFINITION) == 1
    assert combined.count(STRUCTURED_REASONING) == 1
    assert combined.count(DEMONSTRATIONS) == 1
    assert combined.count(COMMON_OUTPUT) == 1
    assert combined.count("Expected output:") == 4
    assert combined.endswith(source)


def test_existing_strategy_prompts_are_unchanged():
    profile = load_profile("reentrancy")
    source = "001 | contract Target {}"
    for strategy, expected_hash in EXISTING_PROMPT_HASHES.items():
        prompt = profile.render_prompt(strategy, source)
        assert hashlib.sha256(prompt.encode()).hexdigest() == expected_hash


def test_all_strategies_receive_identical_preprocessed_target_source():
    profile = load_profile("reentrancy")
    source = "contract T { // remove\n}\n"
    processed = preprocess(source, profile.preprocessing)
    for strategy in (
        "zero_shot",
        "structured_reasoning",
        "few_shot",
        "structured_reasoning_few_shot",
    ):
        assert profile.render_prompt(strategy, processed).endswith(processed)
