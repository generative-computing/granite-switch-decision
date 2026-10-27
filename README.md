# granite-switch-decision

Decision APIs for [Granite Switch](https://github.com/generative-computing/granite-switch)
([`granite-switch-4.1-8b`](https://huggingface.co/ibm-granite/granite-switch-4.1-8b) and
[`-30b`](https://huggingface.co/ibm-granite/granite-switch-4.1-30b) on Hugging Face), as a vLLM
plugin. Ask a question about some text and get back probabilities, from the same server and GPU cache
that already serves `/v1/chat/completions`.

Serving a Switch model takes two vLLM plugins. The standard `granite-switch` plugin
(`register_granite_switch`) registers the model, and this one (`granite_switch_decision`) adds the
decision endpoints on top. The container image has both, and the pip install below sets up both. This
project exists to make the decision endpoints usable now, while we work with the vLLM project on
integrating them (see [Upstream status](#upstream-status)).

- `POST /v1/decisions` follows [OpenAI's Decisions API](https://developers.openai.com/api/docs/guides/decisions)
  for text input, and the official OpenAI SDKs work once you change `base_url`. A few parts of the API aren't
  supported, and those requests get a clear error ([details](#what-v1decisions-doesnt-support)).
- `POST /v1/systemone` follows [jev's system-one API](https://docs.typesafe.ai/api).

Each question is answered with one engine read of the adapter's single-token option labels, so
callers do not need a prompt format, an invocation marker or a label scheme.

With the public Granite Switch checkpoints, results are comparable with those on their model cards.

You can run it as a container or add it to a vLLM install you already have. Start with
[Quick start](#quick-start). To check a deployment or run the tests, see [Testing](#testing).

## Quick start

You need a Linux machine with an NVIDIA GPU. The 8b checkpoint fits on one 80 GB GPU.

### Container

Images are published to GitHub Container Registry as
`ghcr.io/generative-computing/granite-switch-decision` for linux/amd64 and linux/arm64, so one tag
works on both. There are two variants of each release:

| Tag | What it is |
|---|---|
| `latest`, `0.2.0` | Decisions and chat. Use this unless you need speech. |
| `latest-asr`, `0.2.0-asr` | The same, plus the speech model ([`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc)) built in, so `/v1/chat/completions` accepts spoken input (see [Speech input](#speech-input-optional)). About 1 GB larger. |

`latest` and `latest-asr` always point at the newest release of their variant; the version tags stay
fixed, so use one of those when you want a repeatable deployment. The pull is about 9 GB and the image takes
about 30 GB on disk (amd64 figures). You also need Docker and the NVIDIA container toolkit.

```bash
# Download the checkpoint (public, not gated, about 18 GB)
hf download ibm-granite/granite-switch-4.1-8b --local-dir /srv/models/granite-switch-4.1-8b

# Run the image
docker run -d --name gss --gpus all --ipc=host -p 8000:8000 \
    -v /srv/models/granite-switch-4.1-8b:/repository:ro \
    ghcr.io/generative-computing/granite-switch-decision:0.2.0
```

Loading takes a few minutes; follow it with `docker logs -f gss`. `--ipc=host` is vLLM's usual
setting for shared memory, and it matters most when you run with tensor parallelism.

The image is built `FROM vllm/vllm-openai:v0.30.0` (Docker Hub), and adds `granite-switch[audio]==0.1.1`
(the standard Switch plugin), `transformers==5.17.0` and this plugin. Both plugins are enabled for you. It
has no Granite Switch weights in it: it reads the checkpoint from `/repository` (override with
`MODEL_DIR`), so the same image serves the 8b and the 30b. The one exception is the `-asr` image, which
also carries the speech model. We tested the 30b on two 80 GB GPUs; the
container uses every GPU it can see unless you set `TENSOR_PARALLEL_SIZE`.

### Existing vLLM environment

You need Python 3.11 or later and PyTorch 2.11 or later (CUDA 13 builds are tested). **Use vLLM
0.30.0**: it is the version the live tests ran on. granite-switch 0.1.1 accepts `vllm>=0.26,<0.31`,
and `v0.1.0` was also served on 0.26 to 0.29, but nothing since has been checked there.

```bash
pip install "granite-switch[vllm]==0.1.1" "transformers>=5.16"
pip install "git+https://github.com/generative-computing/granite-switch-decision@v0.2.0"
export VLLM_PLUGINS=register_granite_switch,granite_switch_decision
export DECISION_INVOCATION_MARKER="### DECISION"
vllm serve ibm-granite/granite-switch-4.1-8b --served-model-name granite-switch \
    --dtype bfloat16 --attention-config.flash_attn_version=2 --enable-prompt-tokens-details
```

List both plugins in `VLLM_PLUGINS`. vLLM loads endpoint plugins only when they are named, and once
the variable is set it loads nothing else, so leaving out `register_granite_switch` stops the Switch
model class from registering.

### Hugging Face Inference Endpoints

Create an endpoint for the model repository and choose a custom container. Hugging Face's
[custom container guide](https://huggingface.co/docs/inference-endpoints/en/guides/custom_container)
says the model weights are stored under `/repository`, which is where this image looks. The image
serves on port `8000` and answers `/health`, so those are the values for the endpoint's container port
and health route. The guide lists public images and private ones on Docker Hub, AWS ECR, Azure ACR
and Google GCR. GitHub Container Registry is not in that list, but the image is public, so it needs no
registry credentials.

### Check that it works

```bash
curl -fsS http://localhost:8000/health && echo healthy
curl -fsS http://localhost:8000/v1/models         # lists "granite-switch"
python3 scripts/e2e_check.py --base-url http://localhost:8000
```

`e2e_check.py` needs only the standard library and works against any host, so run it from a checkout of
this repository. It exits 0 when the wire formats on both routes, route parity, chat and prefix-cache
sharing all pass. If the server stops at startup, its log names the cause: an unknown
`DECISION_ADAPTER_NAME`, a checkpoint with no `decision` adapter, or an invocation marker that does not
activate the adapter in the chat template.

## Make a request

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

[examples/decisions.md](examples/decisions.md) has the OpenAI SDK, all three question types and the
equivalent `/v1/systemone` request. For equivalent questions both routes build the same prompt and
return the same probabilities.

### More request options

`/v1/systemone` also accepts the extra fields on vLLM's own endpoint, so most clients written for vLLM work as they
are.

- **`cache_salt`** keeps a request's prefix cache separate from everyone else's. The cache is shared, so a fast
  response (or a `cached_tokens` count) can show that someone has already sent the same prompt. If you serve several
  tenants, have your gateway set a long random `cache_salt` for each. It's a privacy measure, not authentication.
  `/v1/decisions` accepts it too.
- **`chat_template_kwargs`** may only be `{"enable_thinking": false}`. The server always chooses the adapter.
- **`joint`, `images`, `audio`, `videos`, `priority`** are accepted at their defaults. This model reads text, one
  question at a time, so other values get a 422.
- **`request_id`** is accepted and ignored.
- **`model`** is required here, though vLLM makes it optional.

Each question in a `/v1/systemone` response also carries `diagnostics`: `label_mass`, how much probability the model
put on the option labels (a low value usually means a wrong marker or layout), and `argmax_is_label`. OpenAI's
response schema doesn't allow extra fields, so `/v1/decisions` has none. `confidence` uses jev's formula, so it isn't
comparable with vLLM's.

## Secure it before you expose it

Check the security of any image you run, including this one and the `vllm/vllm-openai` image it is built
on. For the most control, build your own image from this repository's `Dockerfile` (change the base image
or pinned versions as you need), or install the plugins straight into a vLLM environment you already
maintain, as in [Existing vLLM environment](#existing-vllm-environment).

The server listens on `0.0.0.0` and, as shipped, asks for no credentials on any route. Put it behind a
gateway that authenticates callers, or turn on vLLM's `--api-key`, which `serve.sh` passes through:

```bash
docker run -d --gpus all --ipc=host -p 8000:8000 -v … granite-switch-decision --api-key "$KEY"
```

In vLLM 0.30.0 that key check covers every `/v1` route, including `/v1/decisions` and
`/v1/systemone`, and leaves `/health` open (`vllm/entrypoints/serve/middleware/authenticate.py`). We
have not run it against a live server, so after enabling it, confirm that an unauthenticated `curl` to
`/v1/decisions` returns 401.

If the server also handles chat, set `DECISION_MAX_CONCURRENT_REQUESTS` (8 to 32 is a sensible
range). It is off by default, and without it a burst of decisions slowed chat p50 from 235 ms to
1,088 ms in our load test.

## Speech input (optional)

Speech is a chat feature. The decision routes take text only, and the 8b and 30b checkpoints enable
speech input in their configuration. Loading a speech model slows startup and decisions never use it,
so `serve.sh` switches it off (`--hf-overrides '{"asr_enabled": false}'`). The checkpoint then starts
with nothing else to download, and `/v1/chat/completions` accepts no audio.

To turn it on, set `DECISION_SPEECH_INPUT=1`. The image runs with the Hugging Face Hub offline, so the
speech model
([`ibm-granite/granite-speech-5.0-470m-turboctc`](https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc),
about 1 GB) has to be available. Any one of these works:

- Use an image with the model built in: the `-asr` tag (for example `latest-asr`), or build it yourself:
  `docker build --build-arg WITH_ASR=1 -t granite-switch-decision:asr .` This is why the `-asr` variant
  exists: the images run with the Hub offline, so the speech model has to be inside.
- Mount a Hub cache that holds the model and set `HF_HUB_CACHE` to that directory.
- Set `HF_HUB_OFFLINE=0` so the model downloads at startup.

On startup `serve.sh` checks for the speech model and for the audio libraries (`av`, `scipy`,
`soundfile`, `soxr`), and stops with a message that names whatever is missing. The images already
include the libraries. If you run the plugin in your own environment, install `vllm[audio]`. Without
it the server starts and passes `/health`, then answers every audio request with HTTP 500 "Please
install vllm[audio]".

The `-asr` image bakes in the multilingual release of the speech model; its model card lists the
supported languages. To bake in the earlier English-only revision instead, pass
`--build-arg ASR_REVISION=947f59af40db9791170a0628cf0f3f4812d720f1`.

## Configuration

`serve.sh`, the image's entrypoint, runs `vllm serve` with these settings. Everything after the image
name is passed on to `vllm serve`.

| Environment variable | Default | Effect |
|---|---|---|
| `MODEL_DIR` | `/repository` | Where the checkpoint is. |
| `SERVED_MODEL_NAME` | `granite-switch` | The `model` name clients send. |
| `PORT` | `8000` | Listening port. |
| `MAX_MODEL_LEN` | `32768` | Longest prompt plus output. |
| `GPU_MEMORY_UTILIZATION` | `0.90` | Share of GPU memory vLLM may use. |
| `TENSOR_PARALLEL_SIZE` | every visible GPU | GPUs to split the model across. |
| `DECISION_ADAPTER_NAME` | the only adapter, else `decision` | Which adapter answers. Checked against the checkpoint at startup; an unknown name stops the server. |
| `DECISION_INVOCATION_MARKER` | none (`### DECISION` in the image) | The text the adapter was trained to activate on. It is required: with a wrong marker the chat template activates the adapter in the wrong place without raising an error. |
| `DECISION_SPEECH_INPUT` | `0` | `1` keeps speech input; see [Speech input](#speech-input-optional). |
| `DECISION_MAX_CONCURRENT_REQUESTS` | `0` (no cap) | Decision requests allowed in flight across both routes. Past the cap a request gets 503 at once, so a burst cannot queue ahead of chat. |
| `DECISION_RENDERER` | `granite_switch_decision.renderer:render` | A `module:function` that builds the prompt and label ids, for an adapter trained differently. |
| `DECISION_EXPERIMENTAL_SYSTEM_PROMPT` | `0` | Experimental, may be removed. `1` adds the trained decision-engine system prompt, as used for the model card's published score. It stops chat requests sharing the decision prefix cache. |

Both routes share these settings; nothing is configured per route.

### Build arguments

`GRANITE_SWITCH_VERSION` (default `0.1.1`) is the granite-switch release to install, with its `audio`
extra. `WITH_ASR=1` bakes in the speech model. `ASR_MODEL` and `ASR_REVISION` name the Hub repository
and the commit that is baked in. The default commit is pinned, so a rebuild does not change when the
model repository moves. The checkpoint is never a build argument: you mount it when the container runs.

### Server flags

`serve.sh` passes these non-default flags to vLLM:

| Flag | Why |
|---|---|
| `--dtype bfloat16` | Composed checkpoints carry `torch_dtype: null` ([granite-switch#126](https://github.com/generative-computing/granite-switch/issues/126)), and vLLM would otherwise load float32. |
| `--attention-config.flash_attn_version=2` | Avoids an FA3 schedule bug with CUDA graphs and prefix caching ([granite-switch#139](https://github.com/generative-computing/granite-switch/issues/139), open). With granite-switch 0.1.1 we measured FA3 the same as FA2 for answers and speed, so this is a precaution. |
| `--enable-prompt-tokens-details` | vLLM reports each request's prefix-cache hits, and `/v1/decisions` returns them as `usage.input_tokens_details`. Without the flag both counts are 0. |
| `--max-logprobs 256` | For chat clients that read labels by top-K logprobs (vLLM's default cap is 20). The plugin does not need it. |
| `--enable-auto-tool-choice --tool-call-parser granite4` | Tool calling on the shared chat route. |

There is no server-wide temperature override, so chat requests keep the model's defaults. Prefix
caching stays at vLLM's default, which is on.

On every decision request the plugin itself sets `temperature=0.0`, `max_completion_tokens=1`,
`logprob_token_ids` (the option labels' token ids, read exactly rather than from the top-K) and
`chat_template_kwargs={"adapter_name": <DECISION_ADAPTER_NAME>}`. Callers cannot change them. vLLM
caps `logprob_token_ids` at 128 ids, so a `choice` with more options runs as several requests over the
same prompt, served from the prefix cache.

## Behaviour and limits

- **Prompt.** Two user turns: `State:\n<state>`, then `### DECISION` with the question and lettered
  options (A to Z, then AA, AB and so on past 26), read as a softmax over the label logits. Predicates
  render as "A: No / false" and "B: Yes / true". An option without a description is described by its
  value.
- **No system prompt by default.** A decision can then share the prefix cache with a chat request
  whose first turn is the same `State:` turn. A chat with its own system prompt, or one that sends the
  state without `State:`, shares nothing. Decisions about one state always share with each other. The
  model card's published score used a system prompt, and its effect on accuracy is still being
  evaluated.
- **Caller text is escaped.** Special tokens and the invocation marker inside the state, instructions
  or descriptions are broken with an invisible U+2060, so they cannot forge turns or move the adapter's
  activation point.
- **Input.** `input` is a string or a list of user messages. A message can hold text and `input_json` parts, which
  are written as compact JSON, as an object `state` is on `/v1/systemone`. A question's `instructions` can be a
  string or text and JSON parts. `/v1/systemone` rejects unknown fields and `box` questions.
- **Size.** The state, instructions and options of each question are counted before any engine work:
  at most 400 questions, 32k tokens per question and 64k per request (262k with
  `TypeSafe-Danger-Zone-Long-Context: true` on `/v1/systemone`), all capped by `--max-model-len`.
  Bodies over 32 MiB, nesting deeper than 64 and non-finite numbers are rejected.
- **Usage.** A `choice` with more than 128 options runs as two engine requests over the same prompt,
  and `usage.input_tokens` counts the prompt once per request, so such a question reports about twice
  its prompt length. `cached_tokens` shows how much was shared.
- **Speculative decoding.** Every decision read passes `logprob_token_ids`, which crashes vLLM when a
  speculative-decoding request shares the batch ([vllm#42592](https://github.com/vllm-project/vllm/issues/42592),
  fix [#44727](https://github.com/vllm-project/vllm/pull/44727) open). The plugin logs a warning at
  startup if `speculative_config` is set. We have not tested this combination.
- **Cancellation.** A client that disconnects cancels its request's engine work.
- **Errors.** `/v1/decisions` uses OpenAI's error shape and `/v1/systemone` uses jev's
  `HTTPValidationError` for 422s. Engine errors keep vLLM's status code.
- **Repeat requests.** The same request can return slightly different probabilities, depending on how it is
  batched and whether it hits the prefix cache, so a near-tie can flip. A different GPU or tensor-parallel size can
  differ a little more. vLLM's `VLLM_BATCH_INVARIANT=1` aims to make repeats exact; we haven't tried it here.
- **Not provided.** Authentication (use vLLM's `--api-key`), rate limiting and `refusal` answers.

### What /v1/decisions doesn't support

The decision adapter reads one block of text and a question, and this server keeps to that layout so results stay
predictable. So a few requests that OpenAI's schema allows get a 400 that names the part:

- developer, system and assistant messages (put what the questions are about in a user message)
- tool calls and tool outputs
- images, in `input` or in a question's `instructions`
- choices that would merge into one: two with the same value, or `"true"` and `true`

The route never returns `refusal` answers, and `/v1/systemone` likewise refuses request-level `instructions`.

## Upstream status

This project is temporary. vLLM is adding its own `/v1/systemone`
([#59299](https://github.com/vllm-project/vllm/pull/59299)) and `/v1/decisions`
([#60465](https://github.com/vllm-project/vllm/pull/60465)). Both are on vLLM's `main`, after 0.30.0. We
intend to work with the vLLM project on bringing base Granite Switch support
([#60232](https://github.com/vllm-project/vllm/pull/60232)) and decision support into vLLM itself or a
common plugin, and this package goes away once that is done.

Use it with vLLM 0.30. A vLLM that already has those routes makes the plugin stop at startup, rather than
quietly choosing one, so remove `granite_switch_decision` from `VLLM_PLUGINS` there. The upstream
endpoints differ in some wire details: this plugin follows jev's format and OpenAI's schema as published.

## Testing

The unit and spec tests run in CI, with no GPU. The scripts in `scripts/` check a running server from
outside; they take `--base-url` and use only the standard library unless noted. We have run them on vLLM
0.30.0 with H100 GPUs, serving the 8b and the 30b, mostly on the pip install. Serving the published
images with `docker run` on amd64 hasn't been tried yet.

```bash
pip install -e ".[dev]" && pytest      # unit and spec tests, no GPU
```

| Script | What it checks |
|---|---|
| `e2e_check.py` | Wire formats on both routes, parity, input flattening, errors, usage, vLLM's request fields, prefix-cache sharing |
| `probe_check.py` | Question and size limits, escaping, disconnects; `--max-concurrent` and `--load` add the limiter and a load run |
| `accuracy_check.py` | Accuracy on BoolQ rows (needs internet); `--save` and `--compare` for before/after runs |
| `speech_check.py` | Speech input on a server started with `DECISION_SPEECH_INPUT=1` |
| `load_test.py` | Throughput and latency of `/v1/decisions` at 1 to 128 clients |
| `longctx_check.py` | Long-context caps (needs the tokenizer and a 131k-context server) |

[docs/TESTING.md](docs/TESTING.md) has every test, the commands and options, what runs in CI, and what
needs a tokenizer or a live server.

## Licence

Apache-2.0. Parts adapted from MIT-licensed code are listed, with their notices, in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
