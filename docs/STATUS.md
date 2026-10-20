# Status: what is tested, and what is not

How to run each test is in [TESTING.md](TESTING.md).

Last updated 2026-10-20. "Live" means a real server on H100s: granite-switch 0.1.1,
transformers 5.17, vLLM 0.26.0 / 0.27.1 / 0.28.0 / 0.29.0 / 0.30.0 as noted per row,
and `ibm-granite/granite-switch-4.1-8b` unless a row names another checkpoint.

## Tested

| Area | How | Result |
|---|---|---|
| Unit and route tests | `pytest`, engine stubbed, in CI | pass |
| `/v1/decisions` against OpenAI's OpenAPI schema | `tests/test_openai_spec.py` (42 cases), plus the same request set live | pass, with the exceptions listed in the README |
| `/v1/systemone` against jev's wire format | `tests/test_systemone_spec.py` (24 cases) against `typesafe-sdk`'s generated models, stubbed | pass; the plugin also rejects unknown fields, which jev's format allows |
| Live wire battery | `scripts/e2e_check.py` (11 checks) | 11/11 |
| Prompt rendering | server prompt token counts against an offline render with the checkpoint's tokenizer and chat template | identical |
| Prefix-cache sharing | `tests/test_prefix_sharing.py`, and `cached_tokens` live, cross-checked with `/metrics` | as documented in the README |
| Route parity | 800 labelled items through `/v1/decisions` and through the same prompt over chat completions | no item answers differently |
| Escaping, size guard, disconnect cancellation, error shapes | live probes, now `scripts/probe_check.py` | as documented in the README |
| Question cap and `SYSTEMONE_MAX_CONCURRENT_REQUESTS` | live: 200 questions accepted, 201 rejected; a 32-request burst against a cap of 8 gave 8 accepted and 24 rejected with 503 | pass |
| `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT=1` | live: the 800-item comparison and `e2e_check.py --system-prompt` | answers identical to the same prompt over chat completions; decisions about one state still share the cache, chat no longer does |
| FlashAttention 3 | FA2 and FA3 servers side by side: the 800-item comparison, 40 cached one-token reads (the case in granite-switch#139), load at 8 and 32 clients | no difference beyond batching noise in answers or speed; FA2 kept in `serve.sh` while #139 is open |
| Load | 1 to 128 concurrent clients, three-question requests | about 140 requests/s plateau, no errors |
| Plugin-only install | the README's steps in a fresh environment, then its `vllm serve` command and the e2e battery | pass |
| Install from this repository | `pip install git+https://github.com/generative-computing/granite-switch-systemone` (and the `git+ssh` form) into a fresh venv; entry-point check; the unit and spec suites against the installed copy from a neutral directory | pass, 184 tests; the repository is private, so installing needs repository access |
| vLLM 0.26.0, 0.27.1, 0.28.0, 0.29.0 live | one server per version on one H100 (torch 2.11.0+cu130 on 0.26, torch 2.13.0+cu130 on the rest): the e2e battery, `probe_check`, `test_decisions_e2e`, `test_prefix_sharing`, and the unit suite under each version's own venv | 11/11 e2e and every suite passes on all four versions |
| The 30b checkpoint | `granite-switch-4.1-30b` on vLLM 0.30.0, `--tensor-parallel-size 2` on two H100s: the full battery | pass |
| Tensor parallelism | the 8b with `--tensor-parallel-size 2`: the full battery | pass |
| Long contexts | the 8b with `--max-model-len 131072`, `scripts/longctx_check.py` (9 checks) | 9/9: a 98k-token request answered in one engine read with the danger-zone header and rejected without it; the 131,072 backstop and the default 32k/64k caps enforced; `/v1/decisions` has no header escape |
| Docker image build | `docker build --platform linux/amd64`, plain and `WITH_ASR=1` variants | both build; the stack inside is package-identical to the live-tested one (PyPI vLLM 0.30.0, torch 2.13.0+cu130, transformers 5.17.0, granite-switch 0.1.1); entry points and imports verified in-container |

## Not tested

- **Live serving from the Docker image.** The image builds, and its package stack is
  identical to the live-tested one (the default `vllm/vllm-openai:v0.30.0` tag ships
  the PyPI vLLM wheel with torch 2.13.0+cu130, the same packages the live servers ran),
  but no GPU container host was available to serve from the image itself.
- **The 3b.** The only 3b checkpoint, `granite-switch-4.1-3b-preview`, ships twelve
  adapters (`citations`, `query_rewrite`, `query_clarification`, `hallucination_detection`,
  `answerability`, `factuality-detection`, `policy-guardrails`, `factuality-correction`,
  `guardian-core`, `uncertainty`, `requirement-check`, `context-attribution`), none named
  `decision`, so this plugin refuses to start on it — by design, a decisions server needs
  the decision adapter. No servable 3b exists; the 8b and 30b are the servable released
  checkpoints.
- **The full 262,144-token danger-zone cap.** The cap clamps to `--max-model-len`, and
  every released Switch checkpoint tops out at 131,072 tokens of context; the header path
  is exercised to 131,072 (see the long-contexts row above).
- **Accuracy against the published benchmark.** The checks above compare the routes
  with each other and with a system-prompt variant, not with the model card's Decision
  Index score. A reproduction run through the public Decision Index kit
  (apolinario/decision-index, edition 0.3, fast subset) is staged but blocked on access
  to its one gated source dataset (HLE); the model card's 42.9 (8b) and 50.1 (30b) are
  the numbers to compare against.

## Known gaps

- **Decision load slows chat.** Chat p50 for a 32-token reply went from 235 ms idle to
  1,088 ms with 128 concurrent decision clients and no cap. With
  `SYSTEMONE_MAX_CONCURRENT_REQUESTS=8` and 32 clients it was 362 ms, the excess decision
  requests getting 503s. The cap is off by default, and there is no priority between the
  two kinds of traffic.
- **No system prompt by default.** The model card's published score used one; this plugin
  omits it so decisions can share the prefix cache with chat. Our comparisons found it
  raises accuracy on rating (`score`) questions. `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT=1`
  turns it on (tested live); the default stays off until the trade-off is confirmed.
