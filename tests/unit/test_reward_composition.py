"""Composition of teacher_kl (OPD dense reward) with registry rewards.

Regression: the grpo.py router sent any spec containing "teacher_kl" to
_resolve_teacher_kl alone, silently dropping the other weighted parts
("teacher_kl,ctf_flag*2" trained as pure teacher_kl).
"""

from brainforge.domains.ctf import pack  # noqa: F401  (registers ctf_flag)
from brainforge.training.rewards import compose_reward


def test_compose_with_resolver_routes_teacher_kl():
    calls = []

    def resolver(name: str):
        calls.append(name)
        if name == "teacher_kl":
            return lambda completions, **kw: [2.0 for _ in completions]
        return None

    fn = compose_reward("teacher_kl,ctf_flag*2", resolver=resolver)
    assert calls == ["teacher_kl", "ctf_flag"]
    out = fn(["x", "y"], id=["a", "b"], expected_flag=["F1", "F2"])
    assert out == [2.0, 2.0]  # 2.0 + 2*flag(0) for both


def test_compose_without_resolver_keeps_registry_semantics():
    from brainforge.training import rewards

    rewards.register_reward("_test_plain", lambda completions, **kw: [1.0 for _ in completions])
    fn = compose_reward("_test_plain*0.5")
    assert fn(["x"]) == [0.5]


def test_compose_resolver_none_falls_back_to_registry():
    from brainforge.training import rewards

    rewards.register_reward("_test_fb", lambda completions, **kw: [3.0 for _ in completions])
    fn = compose_reward("_test_fb", resolver=lambda name: None)
    assert fn(["x"]) == [3.0]
