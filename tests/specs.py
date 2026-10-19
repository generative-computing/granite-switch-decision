"""Third-party API schemas for the conformance tests, obtained at test time, never committed.

- OpenAI Decisions: `openapi.json` from openai/openai-openapi (MIT, Copyright (c) OpenAI),
  fetched once at a pinned commit, checked against a pinned SHA-256 and cached outside
  the repository. `SYSTEMONE_OPENAI_SPEC=<path>` uses a local copy instead.
- jev system-one: built from the pydantic models in TypeSafe's MIT-licensed `typesafe-sdk`
  package (a dev dependency). TypeSafe's own OpenAPI document carries no licence, so it
  is not fetched or bundled.

Each loader returns None when the schema is unavailable; the tests skip with `REASON`.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

OPENAI_COMMIT = "33248a638234a53846ed3ab4b2a7c77069bc1030"
OPENAI_URL = f"https://raw.githubusercontent.com/openai/openai-openapi/{OPENAI_COMMIT}/openapi.json"
OPENAI_SHA256 = "c09eea21c178f883549c690752aa082a9fef494a719b7a08ca85341bd033b221"
_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "granite-switch-systemone"

OPENAI_REASON = (
    "OpenAI's OpenAPI spec is unavailable. It is downloaded on first run from "
    f"{OPENAI_URL} (MIT) into {_CACHE}; allow network access once, or set "
    "SYSTEMONE_OPENAI_SPEC to a local copy of that file."
)
JEV_REASON = "jev schemas need TypeSafe's MIT-licensed SDK: pip install typesafe-sdk==0.7.5 (in the dev extra)."


def openai_spec() -> dict | None:
    """OpenAI's full OpenAPI document at the pinned commit, or None if unavailable."""
    local = os.environ.get("SYSTEMONE_OPENAI_SPEC")
    path = Path(local) if local else _CACHE / f"openai-openapi-{OPENAI_COMMIT[:12]}.json"
    if not path.exists() and not local:
        try:
            with urllib.request.urlopen(OPENAI_URL, timeout=60) as r:
                data = r.read()
        except OSError:
            return None
        if hashlib.sha256(data).hexdigest() != OPENAI_SHA256:
            raise RuntimeError(f"{OPENAI_URL} does not match the pinned SHA-256; refusing to use it")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    if not path.exists():
        return None
    data = path.read_bytes()
    if not local and hashlib.sha256(data).hexdigest() != OPENAI_SHA256:
        raise RuntimeError(f"cached {path} does not match the pinned SHA-256; delete it to refetch")
    return json.loads(data)


def openai_validators(*names: str):
    """JSON Schema validators for the named OpenAI component schemas, or None."""
    import jsonschema

    spec = openai_spec()
    if spec is None:
        return None
    defs = json.loads(json.dumps(spec["components"]["schemas"]).replace("#/components/schemas/", "#/$defs/"))
    return [jsonschema.Draft202012Validator({"$ref": f"#/$defs/{n}", "$defs": defs}) for n in names]


def jev_validators(*names: str):
    """JSON Schema validators built from typesafe-sdk's generated wire models, or None.
    These omit the published bounds on choice options (1-255) and score levels (at
    most 10); the plugin's own tests pin those."""
    import jsonschema

    try:
        from typesafe_sdk._schemas import models
    except ImportError:
        return None
    return [jsonschema.Draft202012Validator(getattr(models, n).model_json_schema()) for n in names]
