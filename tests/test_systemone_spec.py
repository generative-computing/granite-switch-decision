# SPDX-License-Identifier: Apache-2.0
"""Conformance of /v1/systemone to jev's wire format, checked with a JSON Schema validator.

The schemas come from the pydantic models in TypeSafe's MIT-licensed `typesafe-sdk` (see
tests/specs.py), which TypeSafe generates from its API spec. They omit the spec's bounds
on choice options (1-255) and score levels (at most 10), so those cases are checked
against the plugin's own limits only. Engine stubbed.

  python3 -m pytest tests/test_systemone_spec.py -v
"""

import pytest

from granite_switch_decision import plugin as sp

pytest.importorskip("jsonschema")
from specs import JEV_REASON, jev_validators  # noqa: E402

_validators = jev_validators("SystemOneRequest", "SystemOneResponse", "HTTPValidationError")
if _validators is None:
    pytest.skip(JEV_REASON, allow_module_level=True)
REQUEST, RESPONSE, ERROR = _validators

N = {"type": "noul", "instructions": "Is it urgent?"}
C = {"type": "choice", "instructions": "Which team?", "criteria": {"payments": "Payout failures", "account": None}}
S = {"type": "score", "instructions": "How urgent?", "criteria": ["can wait", "this week", "today"]}


def _body(**fields):
    return {"model": "granite-switch", "state": "My payouts failed.", "questions": {"q": N}, **fields}


VALID = {
    "string state": _body(),
    "object state": _body(state={"subject": "Duplicate charge", "message": "Please help."}),
    "array state": _body(state=["first message", "second message"]),
    "all three types": _body(questions={"n": N, "c": C, "s": S}),
    "noul with criteria": _body(questions={"n": {**N, "criteria": {"true": "It is urgent", "false": "It can wait"}}}),
    "noul without instructions": _body(questions={"n": {"type": "noul"}}),
    "object instructions": _body(questions={"n": {"type": "noul", "instructions": {"ask": "urgent?"}}}),
    "choice with 1 option": _body(questions={"c": {"type": "choice", "criteria": {"only": "the only one"}}}),
    "choice with 255 options": _body(
        questions={"c": {"type": "choice", "criteria": {f"o{i}": None for i in range(255)}}}
    ),
    "score with 1 level": _body(questions={"s": {"type": "score", "criteria": ["one"]}}),
    "score with 10 levels": _body(questions={"s": {"type": "score", "criteria": [str(i) for i in range(10)]}}),
    "object criteria values": _body(questions={"c": {"type": "choice", "criteria": {"a": {"text": "A"}, "b": ["B"]}}}),
}
# vLLM's own /v1/systemone fields. jev's schema does not have them, so the plugin accepts them as
# an extension (the responses must still be valid jev).
VLLM_FIELDS = {
    "chat_template_kwargs": _body(chat_template_kwargs={"enable_thinking": False}),
    "cache_salt": _body(cache_salt="tenant-1"),
    "joint false": _body(joint=False),
    "empty media": _body(images=[], audio=[], videos=[]),
    "priority 0 and request id": _body(priority=0, request_id="r1"),
}
# Valid in jev's format but rejected on purpose: no image support (`box`), and unknown
# fields, which jev's schema allows and this plugin refuses.
REJECTED = {
    "box question": _body(questions={"x": {"type": "box", "instructions": "Find the cat"}}),
    "request-level instructions": _body(instructions="Be strict."),
    "adapter name in template kwargs": _body(chat_template_kwargs={"adapter_name": "guardian-core"}),
    "unsupported template kwarg": _body(chat_template_kwargs={"documents": []}),
    "thinking on": _body(chat_template_kwargs={"enable_thinking": True}),
    "joint true": _body(joint=True),
    "bad cache salt": _body(cache_salt="a/b"),
    "an image": _body(images=["data:image/png;base64,AA"]),
    "non-zero priority": _body(priority=3),
    "unknown top-level field": _body(temperature=0),
    "unknown question field": _body(questions={"q": {**N, "extra": 1}}),
}
INVALID = {
    "missing state": {"model": "m", "questions": {"q": N}},
    "missing model": {"state": "x", "questions": {"q": N}},
    "no questions": _body(questions={}),
    "score with 0 levels": _body(questions={"s": {"type": "score", "criteria": []}}),
    "null score level": _body(questions={"s": {"type": "score", "criteria": ["a", None]}}),
    "unknown type": _body(questions={"q": {"type": "rank"}}),
}
# Outside the published bounds the SDK models do not carry (choice criteria 1-255, score
# levels at most 10).
OUT_OF_BOUNDS = {
    "choice with 0 options": _body(questions={"c": {"type": "choice", "criteria": {}}}),
    "choice with 256 options": _body(
        questions={"c": {"type": "choice", "criteria": {f"o{i}": None for i in range(256)}}}
    ),
    "score with 11 levels": _body(questions={"s": {"type": "score", "criteria": [str(i) for i in range(11)]}}),
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
        return name, answer, sp.TokenUsage(input_tokens=10, output_tokens=1), sp.QuestionDiagnostics(0.97, True)

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
def test_valid_request_is_answered_with_a_valid_response(client, name):
    body = VALID[name]
    assert not _errors(REQUEST, body), "test case is not valid jev"
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 200, resp.text
    assert not _errors(RESPONSE, resp.json())
    assert set(resp.json()["answers"]) == set(body["questions"])


@pytest.mark.parametrize("name", REJECTED)
def test_deliberately_rejected_request_gets_a_valid_422(client, name):
    body = REJECTED[name]
    assert not _errors(REQUEST, body), "test case is not valid jev"
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 422
    assert not _errors(ERROR, resp.json())


@pytest.mark.parametrize("name", INVALID)
def test_invalid_request_gets_a_valid_422(client, name):
    body = INVALID[name]
    assert _errors(REQUEST, body), "test case is valid jev"
    resp = client.post("/v1/systemone", json=body)
    assert resp.status_code == 422, resp.text
    assert not _errors(ERROR, resp.json())


@pytest.mark.parametrize("name", OUT_OF_BOUNDS)
def test_request_outside_the_published_bounds_gets_a_valid_422(client, name):
    resp = client.post("/v1/systemone", json=OUT_OF_BOUNDS[name])
    assert resp.status_code == 422, resp.text
    assert not _errors(ERROR, resp.json())


@pytest.mark.parametrize("name", VLLM_FIELDS)
def test_vllm_fields_are_accepted_and_the_response_is_still_valid_jev(client, name):
    resp = client.post("/v1/systemone", json=VLLM_FIELDS[name])
    assert resp.status_code == 200, resp.text
    assert not _errors(RESPONSE, resp.json())


def test_response_carries_diagnostics_for_every_question(client):
    body = _body(questions={"n": N, "c": C, "s": S})
    data = client.post("/v1/systemone", json=body).json()
    assert set(data["diagnostics"]) == {"n", "c", "s"}
    assert data["diagnostics"]["n"] == {"label_mass": 0.97, "argmax_is_label": True}
    assert not _errors(RESPONSE, data)


def test_the_salt_reaches_the_engine_options(client, monkeypatch):
    seen = {}
    real = sp._answer_all

    async def spy(state, state_value, questions, options=sp.ReadOptions()):
        seen["options"] = options
        return await real(state, state_value, questions, options)

    monkeypatch.setattr(sp, "_answer_all", spy)
    resp = client.post("/v1/systemone", json=_body(cache_salt="tenant-1"))
    assert resp.status_code == 200
    assert seen["options"] == sp.ReadOptions("tenant-1")
