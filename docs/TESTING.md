# Testing

Results of each check are in [STATUS.md](STATUS.md). This page says what each test
covers and how to run it.

## Automated (CI, every push and pull request)

```bash
pip install -e ".[dev]"
ruff format --check src tests scripts && ruff check src tests scripts
pytest
```

- `tests/test_plugin.py`: request validation, body limits, the context guard, chunked
  label reads and their usage counts, adapter-name resolution, the concurrency limiter,
  disconnect cancellation, startup checks.
- `tests/test_decisions.py`: the `/v1/decisions` wire format with the engine stubbed,
  including parity with `/v1/systemone` (both routes must build the same internal
  question) and error shapes.
- `tests/test_openai_spec.py`: `/v1/decisions` requests and responses validated with a
  JSON Schema validator against OpenAI's published schema. The schema is downloaded on
  first run from `openai/openai-openapi` (MIT) at a pinned commit, checked against a
  pinned SHA-256 and cached in `~/.cache/granite-switch-systemone`; it is never committed.
  Offline, the test skips and says how to supply a local copy (`SYSTEMONE_OPENAI_SPEC`).
  Deliberate limits are listed as named exceptions.
- `tests/test_systemone_spec.py`: `/v1/systemone` validated against jev's wire models from
  TypeSafe's MIT-licensed `typesafe-sdk` (a dev dependency), which TypeSafe generates
  from its API spec. Those models omit the spec's bounds on choice options (1-255) and
  score levels (at most 10), which are checked against the plugin's own limits.
  Without `typesafe-sdk` the test skips and says what to install.
- `tests/test_renderer.py`: prompt layout, label pool, special-token and marker
  escaping, the experimental system prompt switch.

## Manual: needs the checkpoint's tokenizer, no GPU

```bash
pip install "transformers>=5.16"
SYSTEMONE_TOKENIZER=ibm-granite/granite-switch-4.1-8b pytest tests/test_prefix_sharing.py
```

`tests/test_prefix_sharing.py` renders the real prompt with the checkpoint's tokenizer
and chat template and counts which 16-token prefix-cache blocks a decision can share with
other requests. It is skipped in CI because it downloads the tokenizer.

## Manual: needs a running server

Start the server (`serve.sh`, or the commands in the README) on a machine with no other
traffic: several checks read vLLM's server-wide `/metrics`. All scripts use only the
Python standard library unless noted.

| Command | What it checks |
|---|---|
| `python3 scripts/e2e_check.py --base-url URL` | Both routes' answer shapes and parity, input flattening, boolean values, error shapes, usage, chat on the same server, prefix-cache hits. Add `--system-prompt` when the server runs with `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT=1`. |
| `python3 scripts/probe_check.py --base-url URL` | Question cap, size guard (no engine work for oversized requests), body limits, special-token and marker escaping, options without descriptions, disconnect cancellation. |
| `python3 scripts/probe_check.py --base-url URL --max-concurrent N` | Adds the concurrency limit: a burst against `SYSTEMONE_MAX_CONCURRENT_REQUESTS=N` gets 200s and 503s. |
| `python3 scripts/probe_check.py --base-url URL --load 30` | Adds a load test at 1, 8 and 32 clients with chat latency alongside. Reported, not asserted. |
| `SYSTEMONE_BASE_URL=URL pytest tests/test_decisions_e2e.py` | `/v1/decisions` and `/v1/systemone` return the same answers for equivalent requests. |

## Third-party schemas

Nothing in this repository is copied from another project's API specification. See
[THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for what is used and under which
licence. TypeSafe's own `openapi.json` carries no licence and is not used.

## Not in this repository

Accuracy was compared on labelled public datasets (BoolQ, SST-5, Banking77): the
`/v1/decisions` route against the same prompt sent over `/v1/chat/completions`, against
the layout with the system prompt, and FlashAttention 2 against 3. That harness and its
results are not published here; STATUS.md says what was compared.
