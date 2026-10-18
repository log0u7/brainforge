"""Docs coverage guard: every CLI command must be documented.

Prevents the docs drift that already bit this repo once (train commands
documented as stubs while implemented). A new command without a docs entry
fails this test.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]


def _cli_commands() -> set[str]:
    from brainforge.cli import app

    commands = set()
    for group in app.registered_groups:
        commands.add(group.name)
        for command in group.typer_instance.registered_commands:
            commands.add(f"{group.name} {command.name}")
    return commands


def test_every_cli_command_documented():
    docs = (REPO_ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
    missing = []
    for command in sorted(_cli_commands()):
        pattern = rf"(?<![\w-]){re.escape(command)}(?![\w-])"
        if not re.search(pattern, docs):
            missing.append(command)
    assert not missing, f"commands missing from docs/reference/cli.md: {missing}"


def test_cli_md_documents_no_stub_language():
    """The old docs described implemented commands as stubs; keep it fixed."""
    docs = (REPO_ROOT / "docs" / "reference" / "cli.md").read_text(encoding="utf-8")
    assert "not yet implemented" not in docs
    assert "phase 2 stub" not in docs.lower()
