import pytest

from brainforge.training.chat import chat_loop, generate_reply
from brainforge.training.models import load_student_model


class TestChatLoop:
    def test_roundtrip(self):
        history = []
        inputs = iter(["hello", "quit"])
        printed = []
        chat_loop(
            "unused-model-path",
            reply_fn=lambda h: history.append(h) or "ok",
            input_fn=lambda prompt: next(inputs),
            print_fn=lambda text: printed.append(text),
        )
        # reply_fn receives the conversation, loop keeps full history
        assert history == [["hello"]]
        assert any("ok" in line for line in printed)

    def test_history_accumulates(self):
        calls = []

        def reply(history):
            calls.append(list(history))
            return f"reply-{len(calls)}"

        inputs = iter(["one", "two", "quit"])
        chat_loop(
            "unused",
            reply_fn=reply,
            input_fn=lambda prompt: next(inputs),
            print_fn=lambda text: None,
        )
        assert calls[0] == ["one"]
        assert calls[1] == ["one", "two"]

    def test_exit_command(self):
        inputs = iter(["exit"])
        chat_loop(
            "unused",
            reply_fn=lambda h: "x",
            input_fn=lambda p: next(inputs),
            print_fn=lambda t: None,
        )

    def test_eof_exits(self):
        # input_fn raises EOFError like input() on Ctrl+D
        def eof(prompt):
            raise EOFError

        chat_loop("unused", reply_fn=lambda h: "x", input_fn=eof, print_fn=lambda t: None)

    def test_blank_lines_ignored(self):
        replies = []

        def reply(history):
            replies.append(list(history))
            return "ok"

        inputs = iter(["", "  ", "hello", "quit"])
        chat_loop(
            "unused", reply_fn=reply, input_fn=lambda p: next(inputs), print_fn=lambda t: None
        )
        assert replies == [["hello"]]


class TestLoadStudentModel:
    def test_missing_path_fails_cleanly(self, tmp_path):
        from brainforge.errors import BrainforgeError

        with pytest.raises(BrainforgeError, match="not found"):
            load_student_model(tmp_path / "ghost")


def test_generate_reply_signature():
    import inspect

    params = inspect.signature(generate_reply).parameters
    assert list(params) == ["model", "tokenizer", "history", "max_new_tokens"]
