"""GRPO reward function tests - pure string scoring, no torch needed."""

from brainforge.training.rewards import (
    COMPLIANCE_REWARD,
    FORMAT_REWARD,
    FULL_REWARD,
    ZERO_REWARD,
    compliance_reward,
    extract_flag,
    flag_reward,
)


def test_extract_flag_standard_format():
    assert extract_flag("The flag is WQB{s0me_v4lue} end.") == "WQB{s0me_v4lue}"


def test_extract_flag_none_when_absent():
    assert extract_flag("I could not retrieve the flag.") is None


def test_flag_reward_exact_match_full():
    rewards = flag_reward(["flag: WQB{abc_123}"], ["WQB{abc_123}"])
    assert rewards == [FULL_REWARD]


def test_flag_reward_exact_match_case_insensitive():
    rewards = flag_reward(["wqb{abc_123}"], ["WQB{ABC_123}"])
    assert rewards == [FULL_REWARD]


def test_flag_reward_format_only_when_no_ground_truth():
    """Tier-B: ground truth unknown - a parseable flag earns FORMAT_REWARD."""
    rewards = flag_reward(["FCSC{d3adbeef}"], [""])
    assert rewards == [FORMAT_REWARD]


def test_flag_reward_wrong_flag_zero():
    rewards = flag_reward(["WQB{wrong}"], ["WQB{right}"])
    assert rewards == [ZERO_REWARD]


def test_flag_reward_no_flag_zero():
    rewards = flag_reward(["no flag here"], ["WQB{abc}"])
    assert rewards == [ZERO_REWARD]


def test_flag_reward_batch_aligns_with_expected():
    rewards = flag_reward(
        ["got WQB{a1}", "nothing", "got X{b2}"],
        ["WQB{a1}", "WQB{x}", ""],
    )
    assert rewards == [FULL_REWARD, ZERO_REWARD, FORMAT_REWARD]


def test_compliance_reward_penalizes_refusal():
    assert compliance_reward(["I'm sorry, but I cannot help with that."]) == [ZERO_REWARD]


def test_compliance_reward_rewards_answer():
    assert compliance_reward(["Here is the exploit chain: ret2plt."]) == [COMPLIANCE_REWARD]


def test_compliance_reward_catches_ai_tic():
    assert compliance_reward(["As an AI language model, I can't assist."]) == [ZERO_REWARD]


def test_reward_registry_resolves_registered_names():
    from brainforge.training.rewards import get_reward, reward_names

    assert "ctf_flag" in reward_names()
    assert "compliance" in reward_names()
    assert callable(get_reward("ctf_flag"))


def test_reward_registry_unknown_name_raises():
    import pytest

    from brainforge.training.rewards import get_reward

    with pytest.raises(KeyError, match="unknown GRPO reward"):
        get_reward("does_not_exist")


def test_compose_reward_weighted_sum():
    from brainforge.training.rewards import compose_reward

    fn = compose_reward("ctf_flag,compliance*0.5")
    completions = ["WQB{good}"]  # exact flag AND compliant
    assert fn(completions, expected_flag=["WQB{good}"]) == [1.0 + 0.5 * 1.0]


def test_compose_reward_mixed_signals():
    from brainforge.training.rewards import compose_reward

    fn = compose_reward("ctf_flag,compliance*0.5")
    completions = ["I cannot answer."]  # no flag AND refusal
    assert fn(completions, expected_flag=["WQB{good}"]) == [0.0 + 0.5 * 0.0]
