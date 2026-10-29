"""The pilot loop: an LLM reads the challenge material and emits fact
predicates, augure decides the technique from the merged facts, lictor
executes against the local lab target, and the flag read from the
running service lands in an agentic training record.

Division of labor: the LLM pilots (it describes what the source says),
augure decides (rule-verified, provenance attached), lictor executes
(measured offsets, the outcome trailed). The flag is never predicted:
it is read from the service the exploit reached.
"""

import json
import re
from pathlib import Path
from typing import Any

PREDICATE_RE = re.compile(r"^([a-z_][a-z0-9_]*)\((.*)\)\.$")

# Value domains the decision engine enforces: a fact outside its domain
# makes the whole suggest merge fail, so the filter drops the line.
VALUE_DOMAINS: dict[str, set[str]] = {
    "fmtstr_read": {"true", "false"},
    "seccomp": {"true", "false"},
    "function_pointer_on_heap": {"true", "false"},
    "nx": {"true", "false"},
    "pie": {"true", "false"},
    "canary": {"true", "false"},
}

# The predicates the LLM may emit: source-knowledge facts only.
# leaked_address is a measured fact (lictor harvest), not source
# knowledge; the profiler emits the binary facts on its own.
LLM_ALLOWED = {
    "vuln",
    "fmtstr_read",
    "seccomp",
    "allocator",
    "glibc_minor",
    "function_pointer_on_heap",
    "win_function",
}

FACT_PROMPT = """You are feeding an auditable exploitation decision engine.
Read the CTF challenge material below and emit ONLY the fact predicates
you can justify from the source code, one per line, Prolog style:
predicate("value").
Allowed predicates (nothing else: the profiler already emits the
binary facts and the leak addresses are measured separately):
vuln, fmtstr_read, seccomp, allocator, glibc_minor,
function_pointer_on_heap, win_function.
Value domains are strict: fmtstr_read, seccomp and
function_pointer_on_heap take ONLY true or false. vuln takes a
technique keyword (fmtstr, uaf, orw, ret2func, ret2plt, shellcode).
Emit nothing else: no prose, no markdown, no code fences.
Only emit facts the source actually shows.

Challenge material:
{material}"""


def clean_facts(llm_text: str, allowed: set[str] | None = None) -> list[str]:
    """Extract predicate lines from the LLM output, dropping prose and
    fences. With an ``allowed`` whitelist, unknown predicates are dropped
    (the decision engine rejects unknown facts; the LLM drifts). Facts
    whose value sits outside the predicate's domain are dropped too: the
    suggest merge rejects the whole batch otherwise."""
    facts = []
    for line in llm_text.splitlines():
        line = line.strip()
        match = PREDICATE_RE.match(line)
        if not match:
            continue
        if allowed and match.group(1) not in allowed:
            continue
        domain = VALUE_DOMAINS.get(match.group(1))
        if domain and match.group(2).strip('"') not in domain:
            continue
        facts.append(line)
    return facts


def build_trajectory(
    challenge_id: str,
    student_input: str,
    facts_steps: list[str],
    decision: dict,
    outcome: dict,
    model: str,
) -> dict:
    """Agentic-shaped record: user challenge, the three pilot steps as
    assistant/tool turns, a final verdict carrying the flag the run
    read."""
    steps = [
        {
            "thought": "read the challenge material, emit the source-derived facts",
            "action": "facts",
            "action_input": {"predicates": facts_steps},
            "observation": "\n".join(facts_steps),
        },
        {
            "thought": "augure decides from the merged facts",
            "action": "augure_decide",
            "action_input": {"facts": facts_steps},
            "observation": json.dumps(decision, ensure_ascii=False),
        },
        {
            "thought": "lictor builds the payload and runs it against the target",
            "action": "lictor_run",
            "action_input": {"technique": decision.get("technique")},
            "observation": json.dumps(outcome, ensure_ascii=False),
        },
    ]
    messages: list[dict] = [{"role": "user", "content": student_input}]
    for step in steps:
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "thought": step["thought"],
                        "action": step["action"],
                        "action_input": step["action_input"],
                    },
                    ensure_ascii=False,
                ),
            }
        )
        messages.append({"role": "tool", "content": step["observation"]})
    verdict = {
        "verdict": "solved" if outcome.get("flag") else outcome.get("status", "failed"),
        "technique": decision.get("technique"),
        "flag": outcome.get("flag"),
        "source": "lictor_run",
    }
    verdict_json = json.dumps(verdict, ensure_ascii=False, indent=2)
    messages.append({"role": "assistant", "content": verdict_json})
    metadata = {
        "pipeline": "pilot",
        "domain": "agentic",
        "pilot": {"model": model, "decision": decision},
        "flag_verified": bool(outcome.get("flag")),
    }
    return {
        "id": challenge_id,
        "domain": "agentic",
        "messages": messages,
        "metadata": metadata,
    }


def ask_facts(provider: Any, model: str, material: str) -> list[str]:
    """One chat completion: the challenge material in, predicate facts out."""
    from brainforge.providers.base import ChatMessage, ChatRequest

    response = provider.complete(
        ChatRequest(
            messages=[ChatMessage(role="user", content=FACT_PROMPT.format(material=material))],
            temperature=0.2,
        ),
        model,
    )
    return clean_facts(response.content or "", allowed=LLM_ALLOWED)


def run_lictor(workdir: Path, args: list[str], lictor_bin: str = "lictor") -> dict:
    """Run one lictor subcommand in the workdir, parse its JSON output when
    the subcommand emits some, else return the trailing status."""
    import subprocess

    result = subprocess.run(
        [lictor_bin, *args], cwd=workdir, capture_output=True, text=True, timeout=600, check=False
    )
    if result.returncode != 0:
        return {"status": "error", "stderr": result.stderr[-400:]}
    for line in reversed((result.stdout or "").splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                parsed: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return {"status": "done", "stdout": (result.stdout or "")[-400:]}


def run_pilot(
    targets: list[tuple[str, str, str, str]],
    workdir: Path,
    provider: Any,
    model: str,
    lictor_bin: str = "lictor",
    lictor_runner: Any = None,
    output: Path | None = None,
) -> list[dict]:
    """One trajectory per target: LLM facts, augure decide, lictor run.

    ``targets`` entries: (challenge_id, host:port, binary, material).
    ``lictor_runner`` overrides the subprocess runner (tests).
    """
    workdir.mkdir(parents=True, exist_ok=True)
    runner = lictor_runner or (lambda wd, args: run_lictor(wd, args, lictor_bin))
    records: list[dict] = []
    for challenge_id, target, binary, material in targets:
        runner(workdir, ["acquire", target, "--from", f"http://127.0.0.1:8000/{binary}"])
        facts = ask_facts(provider, model, material)
        if facts:
            merged = runner(
                workdir, ["suggest", "target.facts", "\n".join(facts), "-o", "target.facts"]
            )
        else:
            merged = {"status": "no_llm_facts"}
        decision = runner(workdir, ["plan", "target.facts", "-s", "7"])
        runner(
            workdir, ["build", binary, "--facts", "target.facts", "-o", "payload.bin", "-s", "7"]
        )
        outcome = runner(
            workdir,
            [
                "run",
                target,
                "--facts",
                "target.facts",
                "--executor",
                "ronin",
                "--payload",
                "payload.bin",
                "--max-attempts",
                "2",
                "-s",
                "7",
                "--flag-regex",
                "FLAG\\{[^}]+\\}",
            ],
        )
        if merged.get("status") == "error":
            outcome["suggest_error"] = merged.get("stderr")
        record = build_trajectory(
            challenge_id=challenge_id,
            student_input=f"Solve this CTF challenge against the local lab target {binary}.",
            facts_steps=facts,
            decision=decision,
            outcome=outcome,
            model=model,
        )
        records.append(record)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        output.write_text(text, encoding="utf-8")
    return records
