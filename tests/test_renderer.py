"""No-GPU checks for renderer.py: the one adapter-specific piece plugin.py calls out to.

python3 -m pytest tests/test_renderer.py -v -s --tb=short
"""

import pytest

from granite_switch_systemone import renderer


class _FakeTok:
    """Tokenizes single ASCII letters/digits as one token each, two-letter strings as
    two tokens -- just enough for build_label_pool's single-token-label scan."""

    _next_id = 0
    _ids: dict[str, int]

    def __init__(self):
        self._ids = {}

    def __call__(self, text, add_special_tokens=False):
        if len(text) == 1:
            if text not in self._ids:
                self._ids[text] = len(self._ids)
            return {"input_ids": [self._ids[text]]}
        return {"input_ids": [0, 1]}  # multi-char: never a valid single-token label


def test_build_label_pool_returns_single_token_letters():
    tok = _FakeTok()
    pool = renderer.build_label_pool(tok, 3)
    assert [label for label, _ in pool] == ["A", "B", "C"]
    assert len({token_id for _, token_id in pool}) == 3


def test_build_label_pool_raises_if_not_enough_single_token_labels():
    class _AlwaysMultiToken:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [0, 1]}

    with pytest.raises(ValueError, match="single-token"):
        renderer.build_label_pool(_AlwaysMultiToken(), 3)


def test_text_of_passes_strings_through():
    assert renderer.text_of("hello") == "hello"
    assert renderer.text_of(None) == ""


def test_text_of_json_dumps_dict_and_list():
    assert renderer.text_of({"a": 1}) == '{"a": 1}'
    assert renderer.text_of([1, 2]) == "[1, 2]"


def test_render_requires_invocation_marker_env_var(monkeypatch):
    monkeypatch.delenv("SYSTEMONE_INVOCATION_MARKER", raising=False)
    tok = _FakeTok()
    with pytest.raises(RuntimeError, match="SYSTEMONE_INVOCATION_MARKER"):
        renderer.render(tok, "a state", "a question?", ["descA", "descB"])


def test_render_produces_two_user_turns_with_marker_in_the_second(monkeypatch):
    monkeypatch.setenv("SYSTEMONE_INVOCATION_MARKER", "### DECISION")
    tok = _FakeTok()
    msgs, label_ids = renderer.render(tok, "a state", "a question?", ["descA", "descB"])
    assert [m["role"] for m in msgs] == ["user", "user"]
    assert msgs[0]["content"] == "State:\na state"
    assert msgs[1]["content"].count("### DECISION") == 1
    assert len(label_ids) == 2


def test_render_no_system_prompt_ever(monkeypatch):
    """Dropping the system prompt is what lets the state turn's tokens match a plain
    /v1/chat/completions conversation about the same state -- see plugin.py's and
    this module's own docstrings on prefix-cache sharing."""
    monkeypatch.setenv("SYSTEMONE_INVOCATION_MARKER", "### DECISION")
    tok = _FakeTok()
    msgs, _ = renderer.render(tok, "a state", "a question?", ["descA", "descB"])
    assert all(m["role"] != "system" for m in msgs)
