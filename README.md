# granite-switch-systemone

A vLLM endpoint plugin that serves the decision adapter of a
[Granite Switch](https://github.com/generative-computing/granite-switch) checkpoint through
two decision APIs, on the same server and GPU cache as `/v1/chat/completions`:

- `POST /v1/decisions`: [OpenAI's Decisions API](https://developers.openai.com/api/docs/guides/decisions).
  The official OpenAI SDKs work by changing `base_url`.
- `POST /v1/systemone`: [jev's system-one API](https://docs.typesafe.ai/api).

Each question is answered by one engine read of the adapter's single-token option labels,
returned as softmax probabilities, with confidence computed as in jev's examples. Callers need no prompt format, marker or label scheme.

What has and has not been tested is in [docs/STATUS.md](docs/STATUS.md); how to run
the tests is in [docs/TESTING.md](docs/TESTING.md).

## Install

`v0.1.3` is the current release. Pick one path:

- **You already run vLLM:** install the plugin into that environment, below. This is the path tested most.
- **You want a container:** build the image from this repository, further below. It is
  lightly tested (see the note there), so prefer the first path if you can.

### Into an existing vLLM environment (simplest)

Prerequisites:

- Linux with an NVIDIA GPU; the 8b fits on one 80 GB H100.
- Python 3.11+, PyTorch 2.11+ (granite-switch requires it; CUDA 13 builds tested).
- **vLLM 0.30.0 is the tested version.** Use it unless you have a reason not to. granite-switch
  0.1.1 pins `vllm>=0.26,<0.31`, and `v0.1.0` was also served live on 0.26 to 0.29, but changes
  since then have been checked on 0.30.0 only. Treat 0.26 to 0.29 as unverified for this
  version (see [docs/STATUS.md](docs/STATUS.md)).

```bash
pip install "granite-switch[vllm]==0.1.1" "transformers>=5.16"
pip install "git+https://github.com/generative-computing/granite-switch-systemone@v0.1.3"
export VLLM_PLUGINS=register_granite_switch,granite_switch_systemone
export SYSTEMONE_INVOCATION_MARKER="### DECISION"
vllm serve ibm-granite/granite-switch-4.1-8b --served-model-name granite-switch \
    --dtype bfloat16 --attention-config.flash_attn_version=2 --enable-prompt-tokens-details
```

`VLLM_PLUGINS` must name both plugins. vLLM loads endpoint plugins only when they are
listed, and once the variable is set it loads nothing else, so leaving out
`register_granite_switch` stops the Switch model class from registering.

### Docker

Release images are published to GitHub Container Registry as
`ghcr.io/generative-computing/granite-switch-systemone` (linux/amd64 only; about 30 GB, so the first
pull takes a while). `latest` is the newest release and each release also has its version as a tag.
The package is currently private, so pulling needs `docker login ghcr.io` with an account that has
access; until it is made public, build the image yourself (below). On a Linux host with an NVIDIA
GPU, the NVIDIA container toolkit and Docker:

```bash
# 1. Download the checkpoint (public, not gated; about 18 GB)
hf download ibm-granite/granite-switch-4.1-8b --local-dir /srv/models/granite-switch-4.1-8b

# 2. Run the image
docker run -d --name gss --gpus all --ipc=host -p 8000:8000 \
    -v /srv/models/granite-switch-4.1-8b:/repository:ro \
    ghcr.io/generative-computing/granite-switch-systemone:0.1.3
```

To build the image yourself instead, clone the repository at a release tag and run
`docker build --platform linux/amd64 -t granite-switch-systemone .` in it, then use that name in
`docker run`.

Then check it ([Verify the deployment](#verify-the-deployment)). Model loading takes a few
minutes; follow it with `docker logs -f gss`. `--ipc=host` is vLLM's standard setting for
shared memory; it matters most with `--tensor-parallel-size` above 1.

The image is `vllm/vllm-openai:v0.30.0` (the default CUDA 13 tag, whose vLLM wheel and
torch match what granite-switch declares and are the packages the live tests ran)
plus `granite-switch[vllm]==0.1.1`, `transformers==5.17.0` and this plugin. Weights are
not baked in: mount a downloaded checkpoint at `/repository`, or set `MODEL_DIR`.

**Build arguments.** `GRANITE_SWITCH_VERSION` (default `0.1.1`) is the granite-switch release
installed, with its `audio` extra in every variant. `WITH_ASR=1` bakes in the speech model. `ASR_MODEL` and
`ASR_REVISION` name the Hub repository and the commit that is baked in; the default commit is the
one the image was tested with, so a rebuild does not change when the model repository moves. To
bake in a different speech model commit, pass
`--build-arg WITH_ASR=1 --build-arg ASR_REVISION=<commit>`. The checkpoint is never a build
argument: it is mounted when the container runs.

**Speech input.** The 8b and 30b checkpoints enable speech input, and Granite Switch then loads a
speech model while the engine starts. Decisions never use it, so `serve.sh` switches speech
input off (`--hf-overrides '{"asr_enabled": false}'`): the 8b and 30b start from the checkpoint
alone, with nothing else to download or mount. The same server's `/v1/chat/completions` then
takes no audio input.

To keep speech input, set `SYSTEMONE_SPEECH_INPUT=1`. The image runs with the Hub offline, so
the speech model (`ibm-granite/granite-speech-5.0-470m-turboctc`, about 1 GB) must then be
available. Any one of:

- build the image with it baked in (a pinned revision):
  `docker build --platform linux/amd64 --build-arg WITH_ASR=1 -t granite-switch-systemone:asr .`
- mount a Hub cache that holds it and set `HF_HUB_CACHE` to that directory;
- set `HF_HUB_OFFLINE=0` so it downloads at startup.

With `SYSTEMONE_SPEECH_INPUT=1`, `serve.sh` checks for the model first and, if it is missing,
stops with a message naming the model and these options. It also checks that the audio libraries
(`av`, `scipy`, `soundfile`, `soxr`) are installed. The release images include them. If you run
the plugin outside the image, install `vllm[audio]` too: without it the server starts and passes
`/health`, then answers every audio request with HTTP 500 "Please install vllm[audio]".
**Status of the image: lightly tested.** It builds (amd64, with and without `WITH_ASR=1`) and
the packages and plugin entry points were checked inside it. The published `0.1.3` image was served on
H100s with only the checkpoint mounted and speech input off, the 8b on one GPU and the 30b on two with tensor
parallelism, and the wire, robustness and accuracy checks passed. That was run under enroot,
not `docker run` itself; see [docs/STATUS.md](docs/STATUS.md) for what was and was not run. The
pip path above is what most of the live tests used. If you run the image, treat the first
start as the test and report what you find.

## Verify the deployment

```bash
curl -fsS http://localhost:8000/health && echo healthy
curl -fsS http://localhost:8000/v1/models        # lists "granite-switch"
python3 scripts/e2e_check.py --base-url http://localhost:8000    # needs only the standard library
python3 scripts/accuracy_check.py --base-url http://localhost:8000   # answers are still good (needs internet)
```

`e2e_check.py` exits 0 when all its checks pass (wire formats on both routes, route
parity, chat still working, prefix-cache sharing). Run it from a checkout of this
repository; it works against any host. If the server exits at startup, the log names
the cause: an unknown `SYSTEMONE_ADAPTER_NAME`, a checkpoint with no `decision` adapter, or an
invocation marker the chat template does not activate the adapter on (checked at startup).

## Securing the server

The server listens on `0.0.0.0` and, as shipped, requires no authentication on any route.
Do not expose it directly. Either put it behind a gateway that authenticates callers, or
add vLLM's `--api-key`, which `serve.sh` passes through:

```bash
docker run -d --gpus all --ipc=host -p 8000:8000 -v … granite-switch-systemone --api-key "$KEY"
```

vLLM's key check is app-wide middleware that rejects any request under `/v1` without
`Authorization: Bearer <key>` (it reads `vllm/entrypoints/serve/middleware/authenticate.py`
in v0.30.0), so it covers `/v1/decisions` and `/v1/systemone`; `/health` stays open.
Not run against a live server here: after enabling it, confirm an unauthenticated `curl`
to `/v1/decisions` returns 401. Request size limits are under Behaviour and limits. Set `SYSTEMONE_MAX_CONCURRENT_REQUESTS`
(for example 8 to 32) on a server shared with chat: it is off by default, and uncapped decision
bursts slowed chat p50 from 235 ms to 1,088 ms in our load test (see docs/STATUS.md).

## Upstream status

vLLM now has its own `/v1/systemone` ([#59299](https://github.com/vllm-project/vllm/pull/59299),
merged 2026-10-08) and `/v1/decisions` ([#60465](https://github.com/vllm-project/vllm/pull/60465),
merged 2026-10-13). Both are on `main` after 0.30.0, so the next vLLM release will carry them.
This package is for vLLM 0.30 (and, untested since `v0.1.0`, 0.26 to 0.29), where they are absent, and it stops being needed when
you move to a release that has them. The upstream endpoints differ from this plugin in wire
details (this one follows jev's format and OpenAI's schema as published).

The plugin refuses to start if either path is already registered, so running it on a vLLM
that has the upstream endpoints fails at startup instead of silently picking one. Remove
`granite_switch_systemone` from `VLLM_PLUGINS` there.

Still open upstream: native Granite Switch support
([#60232](https://github.com/vllm-project/vllm/pull/60232)) and modular read strategies
([RFC #59365](https://github.com/vllm-project/vllm/issues/59365)). If the RFC lands, the
natural migration is a Granite Switch read strategy; `renderer.render(tok, state,
instructions, descs) -> (messages, label_ids)` is already close to what a strategy needs.

## Server flags

`serve.sh` (the image's entrypoint) runs `vllm serve` with these non-default flags:

| Flag | Why |
|---|---|
| `--dtype bfloat16` | Composed checkpoints carry `torch_dtype: null` ([granite-switch#126](https://github.com/generative-computing/granite-switch/issues/126)); vLLM would otherwise load float32. |
| `--attention-config.flash_attn_version=2` | Avoids an FA3 schedule bug with CUDA graphs and prefix caching ([granite-switch#139](https://github.com/generative-computing/granite-switch/issues/139), open). With granite-switch 0.1.1 we measured FA3 the same as FA2 for answers and speed, so this is a precaution. |
| `--enable-prompt-tokens-details` | vLLM reports each request's prefix-cache hits; `/v1/decisions` returns them as `usage.input_tokens_details`. Without it both counts are 0. |
| `--max-logprobs 256` | For chat clients that read labels by top-K logprobs (vLLM's default cap is 20). The plugin does not need it. |
| `--enable-auto-tool-choice --tool-call-parser granite4` | Tool calling on the shared chat route. |
| `--served-model-name`, `--max-model-len 32768`, `--gpu-memory-utilization 0.90`, `--tensor-parallel-size` | Overridable with `SERVED_MODEL_NAME`, `MAX_MODEL_LEN`, `GPU_MEMORY_UTILIZATION`, `TENSOR_PARALLEL_SIZE`. |

There is no server-wide temperature override, so chat requests keep the model's defaults.
Prefix caching stays at vLLM's default (on).

### What the plugin sets on every decision request

Decisions need a deterministic one-token read, so the plugin sends these itself and
callers cannot change them:

- `temperature=0.0` and `max_completion_tokens=1`
- `logprob_token_ids` set to the option labels' token ids, read exactly rather than from
  the top-K. vLLM caps this at 128 ids, so a `choice` with more options runs as several
  requests over the same prompt, served from the prefix cache.
- `chat_template_kwargs={"adapter_name": <SYSTEMONE_ADAPTER_NAME>}`, which activates the adapter

## Configuration

| Env var | Default | What it does |
|---|---|---|
| `SYSTEMONE_ADAPTER_NAME` | the only adapter, else `decision` | Which adapter answers. Checked against the checkpoint's adapter list at startup; an unknown name stops the server. |
| `SYSTEMONE_INVOCATION_MARKER` | none (`### DECISION` in the image) | The text the adapter was trained to activate on. Required: with a wrong marker the chat template activates the adapter in the wrong place without raising. |
| `SYSTEMONE_RENDERER` | `granite_switch_systemone.renderer:render` | `module:function` that builds the prompt and label ids, for a differently trained adapter. |
| `SYSTEMONE_MAX_CONCURRENT_REQUESTS` | `0` (no cap) | Decision requests in flight on both routes; past it a request gets 503 at once, so decision bursts cannot queue indefinitely ahead of chat. |
| `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT` | `0` | Experimental, may be removed. `1` adds the trained decision-engine system prompt, as used for the model card's published score; it stops chat requests sharing the decision prefix cache. |

Both routes share these; nothing is configured per route.

## Calling it

```bash
curl http://localhost:8000/v1/decisions -H "Content-Type: application/json" -d '{
  "model": "granite-switch",
  "input": "My payouts have failed three times. The bank says everything is fine.",
  "questions": [
    {"type": "choice", "name": "queue", "instructions": "Which team should handle this?",
     "choices": [{"value": "payments", "description": "Payout failures"},
                 {"value": "account", "description": "Login issues"}]},
    {"type": "predicate", "name": "escalate", "instructions": "Does this need urgent attention?"}
  ]}'
```

[examples/decisions.md](examples/decisions.md) has the OpenAI SDK, all three question types
and the equivalent `/v1/systemone` request. Both routes build the same prompt for
equivalent questions and return the same probabilities.

## Behaviour and limits

- **Prompt.** Two user turns, `State:\n<state>` then `### DECISION` with the question and
  lettered options (A–Z, then AA, AB, ... for more than 26), read as a softmax over the
  label logits. Predicates render "A: No / false", "B: Yes / true". An option without a
  description is described by its value.
- **No system prompt by default.** This lets a decision share the prefix cache with a chat request
  whose first turn is the same `State:` turn. Any chat with its own system prompt, or
  that sends the state without `State:`, shares nothing; decisions about one state always
  share it with each other. The model card's published score used a system prompt; its effect
  on accuracy is still being evaluated. `SYSTEMONE_EXPERIMENTAL_SYSTEM_PROMPT=1` turns it on.
- **Caller text is escaped.** Special tokens and the invocation marker inside state,
  instructions or descriptions are broken with an invisible U+2060, so they cannot forge
  turns or move the adapter's activation point.
- **Input.** `input` messages must have role `user`, and images are rejected; the trained
  layout has one text state turn. `/v1/systemone` rejects unknown fields and `box`
  questions.
- **Size.** The state, instructions and options of each question are counted before any
  engine work: at most 200 questions, 32k tokens per question and 64k per request (262k with
  `TypeSafe-Danger-Zone-Long-Context: true` on `/v1/systemone`), capped by
  `--max-model-len`. Bodies over 32 MiB, nesting deeper than 64 and non-finite numbers
  are rejected.
- **Usage.** A `choice` with more than 128 options runs as several engine requests over the same
  prompt (128 labels each, so two requests at most), and `usage.input_tokens` counts the prompt
  once per request, so such a question reports about twice its prompt length. `cached_tokens` shows how much was shared.
- **Speculative decoding.** Every decision read passes `logprob_token_ids`, which vLLM crashes on
  when a speculative-decoding request shares the batch
  ([vllm#42592](https://github.com/vllm-project/vllm/issues/42592), fix
  [#44727](https://github.com/vllm-project/vllm/pull/44727) open). The plugin logs a warning at
  startup if `speculative_config` is set. We have not tested this combination.
- **Cancellation.** A client that disconnects cancels its request's engine work.
- **Errors.** `/v1/decisions` uses OpenAI's error shape, `/v1/systemone` uses jev's
  `HTTPValidationError` for 422s. Engine errors keep vLLM's status code.
- **Not provided.** Authentication (use vLLM's `--api-key`, which covers these routes),
  rate limiting, and `refusal` answers. Engine reads are not bit-identical between
  batched and solo requests, so repeated calls can differ in the third decimal place.
- **vLLM 0.32** mounts its own `/v1/systemone` (newer versions add `/v1/decisions`);
  neither knows Switch adapters, and granite-switch 0.1.1 does not support those
  versions yet.

## Testing

```bash
pip install -e ".[dev]" && pytest                              # unit and spec tests, no GPU
python3 scripts/e2e_check.py --base-url http://localhost:8000   # live wire checks
python3 scripts/probe_check.py --base-url http://localhost:8000 # live robustness checks
```

[docs/TESTING.md](docs/TESTING.md) lists every test, what runs in CI and what needs a
tokenizer or a live server.

## License

Apache-2.0. Parts adapted from MIT-licensed code are listed, with their notices, in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
