# Status: what is tested, and what is not

How to run each test is in [TESTING.md](TESTING.md). "Live" means a real server on H100 GPUs with
granite-switch 0.1.1, transformers 5.17 and vLLM 0.30.0, serving the public
`ibm-granite/granite-switch-4.1-8b` unless a row names another checkpoint.

## Tested

| Area | How | Result |
|---|---|---|
| Both routes against their specs | `/v1/decisions` against OpenAI's published schema (the general-availability revision), with requests generated from it; `/v1/systemone` against jev's wire models | pass, with the exceptions listed in the README |
| Unit and route tests | `pytest` in CI, engine stubbed | pass |
| Live checks, 8b and 30b | `scripts/e2e_check.py` (12 checks), `scripts/probe_check.py` and `scripts/accuracy_check.py` on the 8b (one GPU) and the 30b (two GPUs, tensor parallelism) | pass |
| Prompt and route parity | the server's prompt token counts against an offline render with the checkpoint's tokenizer; the same items through `/v1/decisions` and `/v1/systemone` | identical |
| Prefix-cache sharing | `tests/test_prefix_sharing.py`, and `cached_tokens` live | as described in the README |
| Limits and safety | question cap (400 accepted, 401 rejected), size guard, escaping, disconnect cancellation, `DECISION_MAX_CONCURRENT_REQUESTS` | pass |
| Load | 1 to 128 concurrent clients | no errors; throughput and latency are in the output of `scripts/load_test.py` |
| Long contexts | the 8b with `--max-model-len 131072`, `scripts/longctx_check.py` | pass |
| Install | the README's pip steps in a fresh environment, and from this repository anonymously | pass |
| Container images | the default and `-asr` images build for amd64 and arm64; the 8b and 30b were served from the default image, the 8b with `docker run` on arm64 | pass |
| Speech input | `scripts/speech_check.py` against a server with `DECISION_SPEECH_INPUT=1`, on the pip path and on the arm64 `-asr` image | pass |

## Not tested

- **The `0.2.0` images, served.** They are built and published. The checks above ran on earlier
  images and on the pip path with the same code.
- **Serving the image with `docker run` on amd64.** It was served on amd64 another way, which skips
  the image's `ENTRYPOINT`, `--gpus all` and `--ipc=host`.
- **vLLM other than 0.30.0.** granite-switch 0.1.1 accepts 0.26 to 0.30, and `v0.1.0` was served on
  0.26 to 0.29, but nothing since has been checked there.
- **Contexts beyond 131,072 tokens.** Every released Switch checkpoint stops there.

## Known gaps

- **`/v1/decisions` covers OpenAI's schema for text only.** Images, developer, system and assistant
  messages and tool calls get a 400, and `/v1/systemone` refuses request-level `instructions`. The adapter
  reads one state turn and a question, and nothing else is known to work. The README lists each case.
- **Speculative decoding is untested and may crash the engine.** Decision reads use
  `logprob_token_ids`; vLLM issue #42592 (fix #44727, open) crashes on that whenever a
  speculative-decoding request shares the batch. The plugin warns at startup when
  `speculative_config` is set.
- **Decision load slows chat.** Chat p50 for a 32-token reply went from 235 ms idle to
  1,088 ms with 128 concurrent decision clients and no cap. With
  `DECISION_MAX_CONCURRENT_REQUESTS=8` and 32 clients it was 362 ms, the excess decision
  requests getting 503s. The cap is off by default, and there is no priority between the
  two kinds of traffic.
- **No system prompt by default.** The model card's published score used one; this plugin
  omits it so decisions can share the prefix cache with chat. How it affects accuracy is
  still being evaluated. `DECISION_EXPERIMENTAL_SYSTEM_PROMPT=1` turns it on.
