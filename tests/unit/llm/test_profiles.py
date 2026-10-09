from sb.llm.preprocessing import preprocess
from sb.llm.profiles import load_profile
from sb.llm.profiles.reentrancy import DEMONSTRATIONS, REENTRANCY_DEFINITION, STRUCTURED_REASONING


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


def test_all_strategies_receive_identical_preprocessed_target_source():
    profile = load_profile("reentrancy")
    source = "contract T { // remove\n}\n"
    processed = preprocess(source, profile.preprocessing)
    for strategy in ("zero_shot", "structured_reasoning", "few_shot"):
        assert profile.render_prompt(strategy, processed).endswith(processed)
