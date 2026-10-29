"""The pilot loop: an LLM reads the challenge, emits fact predicates,
augure decides, lictor executes, the flag read from the service lands
in an agentic training record. The LLM never decides the technique and
never guesses the flag: its facts feed the rule engine, the flag comes
from the run."""

import json
from pathlib import Path
from typing import Any


def test_clean_facts_keeps_predicates_and_drops_prose():
    from brainforge.experiments.pilot import clean_facts

    llm_text = """Here are the facts I identified:

```prolog
vuln("fmtstr").
fmtstr_read("true").
seccomp("false").
```

Note that the binary also uses read()."""
    facts = clean_facts(llm_text)
    assert facts == ['vuln("fmtstr").', 'fmtstr_read("true").', 'seccomp("false").']


def test_clean_facts_rejects_malformed_lines():
    from brainforge.experiments.pilot import clean_facts

    llm_text = 'vuln("fmtstr").\ndefinitely not a predicate\nexploit_everything("yes").'
    facts = clean_facts(llm_text, allowed={"vuln", "fmtstr_read", "seccomp"})
    assert facts == ['vuln("fmtstr").']


def test_build_trajectory_is_agentic_shape():
    from brainforge.experiments.pilot import build_trajectory

    record = build_trajectory(
        challenge_id="ctf-lictor-fmtstr-001",
        student_input="Solve this CTF challenge.",
        facts_steps=['vuln("fmtstr").', 'fmtstr_read("true").'],
        decision={"technique": "fmtstr_write", "rank": 1},
        outcome={"status": "flagged", "flag": "FLAG{the_format_is_the_weapon}"},
        model="z-ai/glm-5.3-flash",
    )
    assert record["id"] == "ctf-lictor-fmtstr-001"
    assert record["domain"] == "agentic"
    roles = [m["role"] for m in record["messages"]]
    assert roles[0] == "user" and roles[-1] == "assistant"
    steps = [json.loads(m["content"]) for m in record["messages"] if m["role"] == "assistant"][:-1]
    actions = [s["action"] for s in steps]
    assert actions == ["facts", "augure_decide", "lictor_run"]
    verdict = json.loads(record["messages"][-1]["content"])
    assert verdict["flag"] == "FLAG{the_format_is_the_weapon}"
    assert record["metadata"]["pilot"]["model"] == "z-ai/glm-5.3-flash"
    assert record["metadata"]["flag_verified"] is True


def test_run_pilot_records_trajectory(tmp_path: Any, monkeypatch: Any):
    """End-to-end with fakes: a stub LLM provider and a stub lictor runner
    produce one agentic trajectory per target."""
    import brainforge.experiments.pilot as pilot

    class FakeProvider:
        def __init__(self, replies):
            self.replies = replies
            self.requests = []

        def complete(self, request, model):
            self.requests.append(request)
            from brainforge.providers.base import ChatResponse

            return ChatResponse(content=self.replies.pop(0), provider="fake", model=model)

    provider = FakeProvider(['vuln("fmtstr").\nfmtstr_read("true").'])

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        if args[0] == "plan":
            return {"technique": "fmtstr_write"}
        return {"status": "flagged", "flag": "FLAG{the_format_is_the_weapon}"}

    out = tmp_path / "trajectories.jsonl"
    results = pilot.run_pilot(
        targets=[("ctf-lictor-fmtstr-001", "127.0.0.1:9996", "fmtstr", "source material")],
        workdir=tmp_path / "lictor",
        provider=provider,
        model="z-ai/glm-5.3-flash",
        lictor_bin="lictor",
        lictor_runner=fake_lictor,
        output=out,
    )
    assert len(results) == 1
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["id"] == "ctf-lictor-fmtstr-001"
    assert record["messages"][-1]["role"] == "assistant"


def test_clean_facts_enforces_value_domains():
    """The LLM invents out-of-domain values (fmtstr_read takes true/false
    only): lictor suggest rejects the whole merge, so the filter drops
    them instead of passing them through."""
    from brainforge.experiments.pilot import clean_facts

    llm_text = 'vuln("fmtstr").\nfmtstr_read("printf_input_allows_arbitrary_read").\nseccomp("true").'
    facts = clean_facts(llm_text)
    assert facts == ['vuln("fmtstr").', 'seccomp("true").']


def test_suggest_failure_does_not_abort_the_run():
    """A rejected suggest merge must not erase the outcome: the profiler
    facts alone still decide, lictor still runs."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(
                content='vuln("fmtstr").\nfmtstr_read("bogus").', provider="fake", model=model
            )

    seen = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        seen.append(args[0])
        if args[0] == "suggest":
            return {"status": "error", "stderr": "outside the value domain"}
        if args[0] == "plan":
            return {"technique": "ret2func"}
        return {"status": "flagged", "flag": "FLAG{the_data_decides_ret2plt}"}

    out = tmp_file = Path("/tmp/pilot-test-does-not-abort.jsonl")
    results = pilot.run_pilot(
        targets=[("ctf-lictor-ret2plt-001", "127.0.0.1:9998", "ret2plt", "material")],
        workdir=Path("/tmp/pilot-test-does-not-abort"),
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=out,
    )
    verdict = json.loads(results[0]["messages"][-1]["content"])
    assert verdict["flag"] == "FLAG{the_data_decides_ret2plt}"
    assert "run" in seen
    out.unlink(missing_ok=True)
