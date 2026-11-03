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


def test_clean_facts_normalizes_predicate_prefixed_lines():
    """The GRPO lab-loop student (run-20261030) emits every fact in the
    shape predicate("name", "value"). - the semantics all correct (found
    on the final benchmark run: vuln=fmtstr, allocator=tcache, the whole
    set) but the parser dropped every line, so the run scored 0 before
    any exploit attempt. The shape is a wrap of the fact grammar, not a
    different grammar: normalize it instead of losing the run."""
    from brainforge.experiments.pilot import LLM_ALLOWED, clean_facts

    llm_text = (
        'predicate("vuln", "fmtstr").\n'
        'predicate("function_pointer_on_heap", "true").\n'
        'predicate("allocator", "tcache").\n'
        'predicate("glibc_minor", "15").\n'
        'predicate("fmtstr_read", "false").\n'
        'predicate("seccomp", "false").\n'
    )
    facts = clean_facts(llm_text, allowed=LLM_ALLOWED)
    assert facts == [
        'vuln("fmtstr").',
        'function_pointer_on_heap("true").',
        'allocator("tcache").',
        "glibc_minor(15).",
        'fmtstr_read("false").',
        'seccomp("false").',
    ]


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


def test_run_lictor_timeout_returns_a_failed_record(tmp_path: Any):
    """A hung lictor subcommand (dead service, drip-feed stdout) must not
    kill the whole run: run_lictor turns subprocess.TimeoutExpired into
    a failed record - the caller's timeout path, not an exception that
    destroys a full 11-target run (measured)."""
    import brainforge.experiments.pilot as pilot

    result = pilot.run_lictor(tmp_path, ["5"], lictor_bin="sleep", timeout=1)
    assert result["status"] == "error"
    assert result["timed_out"] is True
    assert "timed out" in result["stderr"]


def test_run_pilot_one_blocked_target_does_not_destroy_the_run(tmp_path: Any):
    """One blocked target (hung runner raising TimeoutExpired) produces a
    failed record; the remaining targets still run and land their flags."""
    import subprocess

    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    call_count = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        call_count.append(args[0])
        if len(call_count) == 1:
            raise subprocess.TimeoutExpired("lictor", 600)
        if args[0] == "plan":
            return {"technique": "ret2func"}
        return {"status": "flagged", "flag": "FLAG{the_data_decides_ret2func}"}

    out = tmp_path / "traj.jsonl"
    results = pilot.run_pilot(
        targets=[
            ("ctf-lictor-blocked-001", "127.0.0.1:9991", "blocked-target", "mat"),
            ("ctf-lictor-ret2func-001", "127.0.0.1:9999", "ret2func", "mat"),
        ],
        workdir=tmp_path / "lictor",
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=out,
    )
    assert len(results) == 2
    first_verdict = json.loads(results[0]["messages"][-1]["content"])
    assert first_verdict["verdict"] == "timeout"
    second_verdict = json.loads(results[1]["messages"][-1]["content"])
    assert second_verdict["flag"] == "FLAG{the_data_decides_ret2func}"


def test_suggest_error_prints_readable_stderr(tmp_path: Any, capsys: Any):
    """The '1 facts -> error' path must show WHY the merge failed, not
    just the word 'error' (measured: the suggest stderr was lost in the
    pilot log)."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        if args[0] == "suggest":
            return {"status": "error", "stderr": "value outside the domain: fmtstr_read(xyz)"}
        if args[0] == "plan":
            return {"technique": "ret2func"}
        return {"status": "flagged", "flag": "FLAG{ok}"}

    out = tmp_path / "traj.jsonl"
    pilot.run_pilot(
        targets=[("ctf-lictor-x-001", "127.0.0.1:9998", "x", "mat")],
        workdir=tmp_path / "lictor",
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=out,
    )
    captured = capsys.readouterr()
    assert "value outside the domain" in captured.out


def test_run_pilot_seed_flows_to_lictor_commands(tmp_path: Any):
    """The seed parameter drives lictor's bandit draws: a rollout sweep
    needs per-rep seeds, a hardcoded 7 would reproduce the same
    trajectory every time (MR6 dataset expansion needs variety)."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    seen_seeds: list[str] = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        if "-s" in args:
            seen_seeds.append(args[args.index("-s") + 1])
        if args[0] == "plan":
            return {"technique": "ret2func"}
        return {"status": "flagged", "flag": "FLAG{seeded}"}

    out = tmp_path / "traj.jsonl"
    pilot.run_pilot(
        targets=[("ctf-lictor-seed-001", "127.0.0.1:9992", "seed-target", "mat")],
        workdir=tmp_path / "lictor",
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=out,
        seed="42",
    )
    assert seen_seeds and all(s == "42" for s in seen_seeds)


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


def test_clean_facts_deduplicates_repeated_lines():
    """The LLM repeats identical lines hundreds of times on hard targets
    (measured: 802 facts on uaf_heap, each `function_pointer_on_heap`
    emitted ~300 times), which poisons the augure fact base and distorts
    the ranking. clean_facts must deduplicate while preserving order."""
    from brainforge.experiments.pilot import clean_facts

    llm_text = "\n".join(['vuln("uaf").'] * 802)
    assert clean_facts(llm_text) == ['vuln("uaf").']

    llm_text = 'vuln("uaf").\n' * 2 + 'seccomp("true").\n' * 3
    assert clean_facts(llm_text) == ['vuln("uaf").', 'seccomp("true").']


def test_clean_facts_caps_output():
    """A degenerate suggest output must not flood the decision engine:
    clean_facts caps the returned fact count at MAX_SUGGESTED_FACTS
    (16), the first facts in emission order win."""
    from brainforge.experiments import pilot

    llm_text = "\n".join(f"glibc_minor({n})." for n in range(40))
    facts = pilot.clean_facts(llm_text, allowed=pilot.LLM_ALLOWED)
    assert len(facts) == pilot.MAX_SUGGESTED_FACTS
    assert facts[0] == "glibc_minor(0)."
    assert facts[-1] == "glibc_minor(15)."


def test_clean_facts_enforces_value_domains():
    """The LLM invents out-of-domain values (fmtstr_read takes true/false
    only): lictor suggest rejects the whole merge, so the filter drops
    them instead of passing them through."""
    from brainforge.experiments.pilot import clean_facts

    llm_text = (
        'vuln("fmtstr").\nfmtstr_read("printf_input_allows_arbitrary_read").\nseccomp("true").'
    )
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

    out = Path("/tmp/pilot-test-does-not-abort.jsonl")
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


def test_run_omits_payload_when_build_found_no_offset(tmp_path: Any):
    """fmtstr exploits need no payload file (the write goes through %n,
    not a return address): the build succeeds without writing one, and
    the run goes out without a --payload instead of pointing at a
    missing file."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(
                content='vuln("fmtstr").\nfmtstr_read("true").', provider="fake", model=model
            )

    seen = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        seen.append(args)
        if args[0] == "build":
            # fmtstr-class: the build succeeds but writes no payload
            # (the write goes through %n, not a return address)
            return {"status": "done"}
        if args[0] == "plan":
            return {"technique": "fmtstr_write"}
        return {"status": "flagged", "flag": "FLAG{the_format_is_the_weapon}"}

    results = pilot.run_pilot(
        targets=[("ctf-lictor-fmtstr-001", "127.0.0.1:9996", "fmtstr", "material")],
        workdir=tmp_path,
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=tmp_path / "trajectories.jsonl",
    )
    run_args = next(a for a in seen if a[0] == "run")
    assert "--payload" not in run_args
    verdict = json.loads(results[0]["messages"][-1]["content"])
    assert verdict["flag"] == "FLAG{the_format_is_the_weapon}"


def test_run_lictor_parses_the_flag_from_the_runner_stdout(tmp_path: Any, monkeypatch: Any):
    """The closed loop prints the truth to stdout (status line + the flag
    the runner's output carries): the parser must lift both into the
    outcome so the trajectory verdict carries the real flag."""
    import sys
    import types

    import brainforge.experiments.pilot as pilot

    fake_subprocess = types.SimpleNamespace(
        run=lambda argv, cwd, capture_output, text, timeout, check: types.SimpleNamespace(
            returncode=0,
            stdout=(
                "run:     1cfb69f6  (interrupted? resume: x)\n"
                "status:  success\n"
                "  ret2func [executed/success]\n"
                "flag:    FLAG{augure_decides_lictor_executes}\n"
            ),
            stderr="",
        )
    )
    monkeypatch.setitem(sys.modules, "subprocess", fake_subprocess)
    outcome = pilot.run_lictor(
        tmp_path, ["run", "127.0.0.1:9999", "--facts", "target.facts"], lictor_bin="lictor"
    )
    assert outcome["status"] == "success"
    assert outcome["flag"] == "FLAG{augure_decides_lictor_executes}"


def test_clean_facts_integer_predicates_stay_bare():
    """glibc_minor takes a bare integer: a quoted "36" is rejected by the
    fact grammar (measured on the uaf_heap run - the whole suggest merge
    failed on it), so the filter unquotes integer-domain values."""
    from brainforge.experiments.pilot import clean_facts

    llm_text = 'vuln("uaf").\nallocator("tcache").\nglibc_minor("36").'
    facts = clean_facts(llm_text)
    assert facts == ['vuln("uaf").', 'allocator("tcache").', "glibc_minor(36)."]


def test_pilot_harvests_the_service_leak(tmp_path: Any):
    """shellcode and orw need the MEASURED leak (leaked_address("input"))
    before build; the demo harvests the service, the pilot must too."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='seccomp("true").', provider="fake", model=model)

    seen = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        seen.append(args[0])
        if args[0] == "plan":
            return {"technique": "orw"}
        if args[0] == "run":
            return {"status": "flagged", "flag": "FLAG{orw_reads_the_flag}"}
        return {"status": "done"}

    pilot.run_pilot(
        targets=[("ctf-lictor-orw-001", "127.0.0.1:9994", "orw", "material")],
        workdir=tmp_path,
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=tmp_path / "t.jsonl",
    )
    assert "harvest" in seen
    assert seen.index("harvest") < seen.index("build")


def test_run_flag_regex_is_the_target_expected_flag(tmp_path: Any):
    """A generic FLAG{...} oracle matches the FIRST flag in the output -
    the lab's cat prints six. The pilot's run oracle must be the
    target's expected flag (the enriched case carries it), escaped."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    seen = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        seen.append(args)
        if args[0] == "plan":
            return {"technique": "ret2plt"}
        return {"status": "done"}

    pilot.run_pilot(
        targets=[
            (
                "ctf-lictor-ret2plt-001",
                "127.0.0.1:9998",
                "ret2plt",
                "material",
                "FLAG{the_data_decides_ret2plt}",
            )
        ],
        workdir=tmp_path,
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=tmp_path / "t.jsonl",
    )
    run_args = next(a for a in seen if a[0] == "run")
    regex_index = run_args.index("--flag-regex")
    assert run_args[regex_index + 1] == "FLAG\\{the_data_decides_ret2plt\\}"


def test_build_receives_the_decided_technique(tmp_path: Any):
    """The build re-plans with its own bandit draw: without --technique
    the same seed could build the wrong technique's layout (measured:
    the shellcode build drew ret2libc at seeds 7-9, whose layout
    demands a pop_rdi the binary's facts lack). The plan's decided
    technique must ride --technique onto the build."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    seen = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        seen.append(args)
        if args[0] == "plan":
            return {
                "status": "done",
                "stdout": "selected:   shellcode\nranking:    shellcode > ret2libc",
            }
        if args[0] == "build":
            (workdir / "payload.bin").write_bytes(b"\xeb\x16")
            return {"status": "done"}
        if args[0] == "run":
            return {"status": "flagged", "flag": "FLAG{nx_off_says_shellcode}"}
        return {"status": "done"}

    pilot.run_pilot(
        targets=[("ctf-lictor-shellcode-001", "127.0.0.1:9997", "shellcode", "material")],
        workdir=tmp_path,
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=tmp_path / "t.jsonl",
    )
    build_args = next(a for a in seen if a[0] == "build")
    assert "--technique" in build_args
    assert build_args[build_args.index("--technique") + 1] == "shellcode"


def test_build_walks_the_ranking_when_the_draw_is_unbuildable(tmp_path: Any):
    """The plan's ranking IS the decision: a draw whose layout the facts
    cannot supply (measured: ret2libc drawn on the shellcode binary,
    whose facts carry no pop_rdi_ret) must not burn the target - the
    build walks the ranking and the first buildable technique runs."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    builds: list[list[str]] = []
    calls: list[list[str]] = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        calls.append(args)
        if args[0] == "plan":
            return {
                "status": "done",
                "stdout": "selected:   ret2libc\nranking:    ret2libc > shellcode > ret2func",
            }
        if args[0] == "build":
            builds.append(args)
            technique = args[args.index("--technique") + 1]
            if technique == "ret2libc":
                return {"status": "error", "stderr": "lictor: no pop_rdi_ret gadget in the facts"}
            (workdir / "payload.bin").write_bytes(b"\xeb\x16")
            return {"status": "done"}
        if args[0] == "run":
            return {"status": "flagged", "flag": "FLAG{ranked_walk}"}
        return {"status": "done"}

    records = pilot.run_pilot(
        targets=[("ctf-lictor-shellcode-001", "127.0.0.1:9997", "shellcode", "material")],
        workdir=tmp_path,
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=tmp_path / "t.jsonl",
    )
    assert len(builds) == 2
    assert builds[0][builds[0].index("--technique") + 1] == "ret2libc"
    assert builds[1][builds[1].index("--technique") + 1] == "shellcode"
    run_args = next(a for a in calls if a[0] == "run")
    assert "--payload" in run_args
    verdict = records[0]["messages"][-1]["content"]
    assert "ranked_walk" in verdict


fake_lictor_calls: list[list[str]] = []


def test_payload_less_run_survives_a_fully_failed_ranking(tmp_path: Any):
    """No ranked technique builds: fmtstr-class and heap exploits need
    no payload (the write goes through %n, the heap through the
    allocator) - the run still goes out and its own recipe decides; the
    failed builds stay in the outcome (zero-reward rollouts for the
    trainer)."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    fake_lictor_calls.clear()

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        fake_lictor_calls.append(args)
        if args[0] == "plan":
            return {
                "status": "done",
                "stdout": "selected:   ret2libc\nranking:    ret2libc > shellcode",
            }
        if args[0] == "build":
            return {"status": "error", "stderr": "lictor: no gadget in the facts"}
        if args[0] == "run":
            return {"status": "flagged", "flag": "FLAG{payload_less}"}
        return {"status": "done"}

    records = pilot.run_pilot(
        targets=[("ctf-lictor-fmtstr-001", "127.0.0.1:9996", "fmtstr", "material")],
        workdir=tmp_path,
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=tmp_path / "t.jsonl",
    )
    run_args = next(a for a in fake_lictor_calls if a[0] == "run")
    assert "--payload" not in run_args
    verdict = json.loads(records[0]["messages"][-1]["content"])
    assert verdict["flag"] == "FLAG{payload_less}"
    observation = json.loads(records[0]["messages"][-2]["content"])
    assert [f["technique"] for f in observation["build_failures"]] == ["ret2libc", "shellcode"]


def test_run_executes_the_built_technique(tmp_path: Any):
    """The run's own bandit draw could pick a technique whose exploit
    recipe differs from the payload the build just produced (measured:
    the shellcode run drew ret2libc, no exploit exists for it ->
    handoff x4) - the run receives --technique with the built one."""
    import brainforge.experiments.pilot as pilot
    from brainforge.providers.base import ChatResponse

    class FakeProvider:
        def complete(self, request, model):
            return ChatResponse(content='vuln("sof").', provider="fake", model=model)

    seen: list[list[str]] = []

    def fake_lictor(workdir: Path, args: list[str]) -> dict:
        seen.append(args)
        if args[0] == "plan":
            return {
                "status": "done",
                "stdout": "selected:   shellcode\nranking:    shellcode > ret2libc",
            }
        if args[0] == "build":
            (workdir / "payload.bin").write_bytes(b"\xeb\x16")
            return {"status": "done"}
        if args[0] == "run":
            return {"status": "flagged", "flag": "FLAG{run_pinned}"}
        return {"status": "done"}

    records = pilot.run_pilot(
        targets=[("ctf-lictor-shellcode-001", "127.0.0.1:9997", "shellcode", "material")],
        workdir=tmp_path,
        provider=FakeProvider(),
        model="m",
        lictor_runner=fake_lictor,
        output=tmp_path / "t.jsonl",
    )
    run_args = next(a for a in seen if a[0] == "run")
    assert run_args[run_args.index("--technique") + 1] == "shellcode"
    verdict = json.loads(records[0]["messages"][-1]["content"])
    assert verdict["flag"] == "FLAG{run_pinned}"
