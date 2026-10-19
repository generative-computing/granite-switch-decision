# Status: what is tested, and what is not

How to run each test is in [TESTING.md](TESTING.md).

Last updated 2026-10-19. "Live" means a real server: vLLM 0.30.0, granite-switch 0.1.1,
transformers 5.17, `ibm-granite/granite-switch-4.1-8b`, one H100.

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

## Not tested

- **The Docker image does not build.** The vLLM 0.30 base image ships torch 2.14.0, but
  its own vLLM wheel requires torch 2.13.0, so installing `granite-switch[vllm]` against
  the image's packages has no solution (checked on the amd64 and arm64 images). The 0.26
  base resolves but has not been run. To do: pin the base image's torch stack to what
  vLLM declares, or base the image on a plain CUDA image plus the tested pip install.
  Until then, use the plugin-only install.
- **Installing from this repository.** The plugin-only install was verified from the
  earlier development fork; `generative-computing/granite-switch-systemone` is private,
  so installing from it needs repository access.
- **vLLM 0.26 to 0.29.** Every live run used 0.30.0. The code paths that differ by
  version (where vLLM keeps `ErrorResponse`) are unit-tested with stand-in modules only.
- **Other checkpoints.** Only the 8b. The 3b and 30b have the same adapter layout but have
  not been served with this plugin.
- **Multi-GPU and tensor parallelism.**
- **Long contexts.** Probes stayed under the 32,768-token `--max-model-len`; the 262k
  long-context header path is not exercised.
- **Accuracy against the published benchmark.** The checks above compare the routes
  with each other and with a system-prompt variant, not with the model card's score.

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
