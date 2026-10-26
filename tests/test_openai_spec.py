# SPDX-License-Identifier: Apache-2.0
"""Conformance of /v1/decisions to OpenAI's published OpenAPI schema (general availability,
2026-10-22 and later), checked with a JSON Schema validator. The schema is fetched at test
time from openai/openai-openapi (MIT) at a pinned commit and hash; see tests/specs.py. The
test skips, saying how to supply it, when it cannot be fetched.

Every spec-valid request must answer 200 and every 200 body must validate against
`DecisionResponse`, except the deliberate limits listed in `UNSUPPORTED`, which must
answer 400 with OpenAI's error shape. Spec-invalid requests must answer 400, except the
leniencies in `ACCEPTED_LENIENTLY`.

  python3 -m pytest tests/test_openai_spec.py -v
"""

import pytest

from granite_switch_decision import plugin as sp

pytest.importorskip("jsonschema")
from specs import OPENAI_REASON, openai_validators  # noqa: E402

_validators = openai_validators("DecisionRequest", "DecisionResponse")
if _validators is None:
    pytest.skip(OPENAI_REASON, allow_module_level=True)
REQUEST, RESPONSE = _validators

PREDICATE = {"type": "predicate", "instructions": "Is it urgent?"}
CHOICE = {
    "type": "choice",
    "name": "queue",
    "instructions": "Which team?",
    "choices": [{"value": "a", "description": "Team A"}, {"value": "b"}],
}
SCORE = {
    "type": "score",
    "name": "severity",
    "instructions": "How urgent?",
    "levels": [{"label": "low"}, {"label": "high", "description": "Same day"}],
}


def _body(**fields):
    return {"model": "m", "input": "The export failed.", "questions": [PREDICATE], **fields}


def _message(role, content, **fields):
    return {"role": role, "content": content, **fields}


VALID = {
    "string input": _body(),
    "empty string input": _body(input=""),
    "empty message list": _body(input=[]),
    "all three question types": _body(questions=[PREDICATE, CHOICE, SCORE]),
    "400 questions": _body(questions=[PREDICATE] * 400),
    "empty instructions": _body(questions=[{**PREDICATE, "instructions": ""}]),
    "named predicate": _body(questions=[{**PREDICATE, "name": "urgent"}]),
    "boolean choice values": _body(questions=[{**CHOICE, "choices": [{"value": True}, {"value": False}]}]),
    "255 choices": _body(questions=[{**CHOICE, "choices": [{"value": f"v{i}"} for i in range(255)]}]),
    "10 score levels": _body(questions=[{**SCORE, "levels": [{"label": str(i)} for i in range(10)]}]),
    "user message, string content": _body(input=[_message("user", "hi")]),
    "user message, explicit type": _body(input=[_message("user", "hi", type="message")]),
    "user message, input_text parts": _body(
        input=[_message("user", [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}])]
    ),
    "input_json object part": _body(
        input=[_message("user", [{"type": "input_json", "data": {"ticket": {"body": "Refund me."}, "n": [1, 2]}}])]
    ),
    "input_json array of objects": _body(
        input=[_message("user", [{"type": "input_json", "data": [{"a": 1}, {"b": None}]}])]
    ),
    "text and JSON parts": _body(
        input=[
            _message("user", [{"type": "input_text", "text": "Evaluate."}, {"type": "input_json", "data": {"k": 1}}])
        ]
    ),
    "predicate with instruction parts": _body(
        questions=[
            {
                **PREDICATE,
                "instructions": [{"type": "input_text", "text": "Urgent?"}, {"type": "input_json", "data": {"a": 1}}],
            }
        ]
    ),
    "choice with a JSON instruction part": _body(
        questions=[{**CHOICE, "instructions": [{"type": "input_json", "data": [{"policy": "refunds"}]}]}]
    ),
    "input_text with a cache breakpoint": _body(
        input=[
            _message("user", [{"type": "input_text", "text": "a", "prompt_cache_breakpoint": {"mode": "explicit"}}])
        ]
    ),
    "prompt cache and safety fields": _body(
        prompt_cache_key="k", prompt_cache_options={"ttl": "30m", "mode": "implicit"}, safety_identifier="u1"
    ),
    "null optional fields": _body(prompt_cache_key=None, prompt_cache_options=None, safety_identifier=None),
}

# Spec-valid requests this text decision adapter rejects on purpose (README, "What
# /v1/decisions doesn't support"). Each must fail with a 400 in OpenAI's error shape.
UNSUPPORTED = {
    "system message": _body(input=[_message("system", "Be terse."), _message("user", "hi")]),
    "developer message with text parts": _body(
        input=[_message("developer", [{"type": "input_text", "text": "Be terse."}]), _message("user", "hi")]
    ),
    "assistant message": _body(input=[_message("user", "hi"), _message("assistant", "hello")]),
    "assistant message with output_text and refusal parts": _body(
        input=[
            _message("user", "hi"),
            _message("assistant", [{"type": "output_text", "text": "hello"}, {"type": "refusal", "refusal": "no"}]),
        ]
    ),
    "function call and output": _body(
        input=[
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "42"},
        ]
    ),
    "custom tool call and output": _body(
        input=[
            {"type": "custom_tool_call", "call_id": "c1", "name": "t", "input": "x"},
            {"type": "custom_tool_call_output", "call_id": "c1", "output": [{"type": "input_text", "text": "y"}]},
        ]
    ),
    "image part": _body(input=[_message("user", [{"type": "input_image", "image_url": "https://x.test/y.png"}])]),
    "duplicate choice values": _body(questions=[{**CHOICE, "choices": [{"value": "a"}, {"value": "a"}]}]),
    "image part with detail": _body(
        input=[_message("user", [{"type": "input_image", "image_url": "data:image/png;base64,AA", "detail": "low"}])]
    ),
    "image in question instructions": _body(
        questions=[{**PREDICATE, "instructions": [{"type": "input_image", "image_url": "https://x.test/y.png"}]}]
    ),
    "image in a tool output": _body(
        input=[
            {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"},
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": [{"type": "input_image", "image_url": "https://x.test/y.png"}],
            },
        ]
    ),
}

INVALID = {
    "missing model": {"input": "x", "questions": [PREDICATE]},
    "missing input": {"model": "m", "questions": [PREDICATE]},
    "no questions": _body(questions=[]),
    "401 questions": _body(questions=[PREDICATE] * 401),
    "1 choice": _body(questions=[{**CHOICE, "choices": [{"value": "a"}]}]),
    "256 choices": _body(questions=[{**CHOICE, "choices": [{"value": f"v{i}"} for i in range(256)]}]),
    "1 score level": _body(questions=[{**SCORE, "levels": [{"label": "a"}]}]),
    "11 score levels": _body(questions=[{**SCORE, "levels": [{"label": str(i)} for i in range(11)]}]),
    "predicate without instructions": _body(questions=[{"type": "predicate"}]),
    "numeric choice values": _body(questions=[{**CHOICE, "choices": [{"value": 1}, {"value": 2}]}]),
    "level without label": _body(questions=[{**SCORE, "levels": [{"description": "a"}, {"description": "b"}]}]),
    "input_json with string data": _body(input=[_message("user", [{"type": "input_json", "data": "x"}])]),
    "input_json with an array of numbers": _body(input=[_message("user", [{"type": "input_json", "data": [1, 2]}])]),
    "empty instructions array": _body(questions=[{**PREDICATE, "instructions": []}]),
    "cache breakpoint in an instruction part": _body(
        questions=[
            {
                **PREDICATE,
                "instructions": [{"type": "input_text", "text": "x", "prompt_cache_breakpoint": {"mode": "explicit"}}],
            }
        ]
    ),
    "unknown top-level field": _body(temperature=0),
    "unknown question field": _body(questions=[{**PREDICATE, "extra": 1}]),
    "unknown message role": _body(input=[_message("tool", "x")]),
    "unknown cache breakpoint mode": _body(
        input=[_message("user", [{"type": "input_text", "text": "a", "prompt_cache_breakpoint": {"mode": "x"}}])]
    ),
}

# Spec-invalid requests the plugin answers anyway: the fields are accepted and ignored,
# or a missing name is treated as one, so tightening them would only break clients.
# `input_json` over OpenAI's stated size limits is also accepted; this server's own limits apply.
ACCEPTED_LENIENTLY = {
    "null question name": _body(questions=[{**PREDICATE, "name": None}]),
    "unsupported cache ttl": _body(prompt_cache_options={"ttl": "1h"}),
    "cache key over 64 characters": _body(prompt_cache_key="k" * 65),
}


def _errors(validator, instance):
    return [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator.iter_errors(instance)]


@pytest.fixture()
def client(monkeypatch):
    async def fake_answer(name, q, state, chat, tok, served_model, adapter_name, renderer, reads=None, options=None):
        if q.type == "choice":
            keys = list(q.criteria)
            answer = {
                "type": "choice",
                "choice": keys[0],
                "probabilities": {k: 1.0 / len(keys) for k in keys},
                "confidence": 0.5,
            }
        elif q.type == "noul":
            answer = {"type": "noul", "noul": 0.9}
        else:
            n = len(q.criteria)
            answer = {
                "type": "score",
                "score": 1.0,
                "legend": {str(i): d for i, d in enumerate(q.criteria)},
                "probabilities": {str(i): 1.0 / n for i in range(n)},
                "confidence": 0.3,
            }
        return (
            name,
            answer,
            sp.TokenUsage(input_tokens=10, output_tokens=1, cached_tokens=4),
            sp.QuestionDiagnostics(0.97, True),
        )

    monkeypatch.setattr(sp, "_answer_question", fake_answer)

    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()
    app.include_router(sp.router)
    app.state.decision_chat = object()
    app.state.decision_tokenizer = lambda text, add_special_tokens=False: {"input_ids": [0] * len(str(text).split())}
    app.state.decision_max_model_len = 8192
    app.state.decision_served_model = "granite-switch"
    app.state.decision_adapter_name = "decision"
    app.state.decision_renderer = lambda *args: ([], [])
    app.state.decision_limiter = sp._Limiter(0)
    return TestClient(app)


@pytest.mark.parametrize("name", VALID)
def test_spec_valid_request_is_answered_with_a_spec_valid_response(client, name):
    body = VALID[name]
    assert not _errors(REQUEST, body), "test case is not spec-valid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 200, resp.text
    assert not _errors(RESPONSE, resp.json())


@pytest.mark.parametrize("name", UNSUPPORTED)
def test_unsupported_input_is_rejected_with_openai_error_shape(client, name):
    body = UNSUPPORTED[name]
    assert not _errors(REQUEST, body), "test case is not spec-valid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("name", INVALID)
def test_spec_invalid_request_is_rejected(client, name):
    body = INVALID[name]
    assert _errors(REQUEST, body), "test case is not spec-invalid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 400, resp.text
    assert set(resp.json()["error"]) == {"message", "type", "param", "code"}


@pytest.mark.parametrize("name", ACCEPTED_LENIENTLY)
def test_lenient_acceptance_still_returns_a_spec_valid_response(client, name):
    body = ACCEPTED_LENIENTLY[name]
    assert _errors(REQUEST, body), "test case is not spec-invalid"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 200, resp.text
    assert not _errors(RESPONSE, resp.json())


# vLLM's own /v1/decisions takes these two fields beyond the schema (cache_salt, joint).
VLLM_EXTENSIONS = {
    "cache salt": _body(cache_salt="tenant-1"),
    "joint false": _body(joint=False),
}


@pytest.mark.parametrize("name", VLLM_EXTENSIONS)
def test_vllm_extension_is_accepted_with_a_spec_valid_response(client, name):
    body = VLLM_EXTENSIONS[name]
    assert _errors(REQUEST, body), "test case is spec-valid, so it is not an extension"
    resp = client.post("/v1/decisions", json=body)
    assert resp.status_code == 200, resp.text
    assert not _errors(RESPONSE, resp.json())


def test_the_response_has_no_diagnostics_because_the_schema_forbids_extra_fields(client):
    data = client.post("/v1/decisions", json=_body()).json()
    assert "diagnostics" not in data and not _errors(RESPONSE, data)


@pytest.mark.parametrize("field", ["instructions", "chat_template_kwargs", "priority", "request_id"])
def test_systemone_only_fields_are_rejected_on_decisions(client, field):
    value = {"chat_template_kwargs": {}, "priority": 0}.get(field, "x")
    resp = client.post("/v1/decisions", json=_body(**{field: value}))
    assert resp.status_code == 400
    assert resp.json()["error"]["type"] == "invalid_request_error"


def test_usage_reports_cached_tokens_summed_over_questions(client):
    resp = client.post("/v1/decisions", json=_body(questions=[PREDICATE, CHOICE, SCORE]))
    usage = resp.json()["usage"]
    assert usage["input_tokens_details"] == {"cached_tokens": 12, "cache_write_tokens": 0}
    assert usage["output_tokens_details"] == {"reasoning_tokens": 0}
