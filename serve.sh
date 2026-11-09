#!/bin/bash
# vLLM's OpenAI-compatible server for a composed granite-switch checkpoint at $MODEL_DIR,
# with /v1/decisions and /v1/systemone mounted by the endpoint plugin. Extra container
# arguments are passed to `vllm serve`.
#   MODEL_DIR               default /repository (HF Inference Endpoints' mount point)
#   TENSOR_PARALLEL_SIZE    default: every visible GPU
#   MAX_MODEL_LEN (32768), GPU_MEMORY_UTILIZATION (0.90), PORT (8000), SERVED_MODEL_NAME
#   STARTUP_MODE (default), CUDA_GRAPH_SIZES, QUANT (off), CACHE_DIR (/cache): optional startup
#                           tuning; see README, "Fast startup". All default to today's behaviour.
#   DECISION_ADAPTER_NAME  default "decision"; DECISION_INVOCATION_MARKER, see README
#   DECISION_SPEECH_INPUT  default 0: speech input is switched off, so no speech model is needed.
#                           1 keeps the checkpoint's speech input; see README, "Speech input".
set -e
MODEL_DIR=${MODEL_DIR:-/repository}
# The 8b and 30b checkpoints enable speech input (asr_enabled) and granite-switch then loads a
# speech model at startup, which decisions never use. Off by default; with it on, fail early,
# with the cause, if the speech model is not in the offline cache.
case "${DECISION_SPEECH_INPUT:-0}" in
    0) SPEECH_ARGS=(--hf-overrides '{"asr_enabled": false}') ;;
    1) SPEECH_ARGS=(); python3 -m granite_switch_decision.preflight "$MODEL_DIR" ;;
    *) echo "DECISION_SPEECH_INPUT must be 0 or 1, not '$DECISION_SPEECH_INPUT'" >&2; exit 2 ;;
esac
TP=${TENSOR_PARALLEL_SIZE:-$(python3 -c "import torch; print(max(torch.cuda.device_count(), 1))")}
export DECISION_ADAPTER_NAME=${DECISION_ADAPTER_NAME:-decision}

# Startup tuning. Every option defaults to today's behaviour; a value that is set but not one
# of the listed ones stops the server rather than silently serving with unexpected settings.
STARTUP_ARGS=()
case "${STARTUP_MODE:-}" in
    ""|default) ;;
    fast) STARTUP_ARGS+=(--optimization-level 1) ;;
    eager) STARTUP_ARGS+=(--enforce-eager) ;;
    *) echo "STARTUP_MODE must be default, fast or eager, not '$STARTUP_MODE'" >&2; exit 2 ;;
esac
if [ -n "${CUDA_GRAPH_SIZES:-}" ]; then
    if ! [[ "$CUDA_GRAPH_SIZES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
        echo "CUDA_GRAPH_SIZES must be comma-separated batch sizes, not '$CUDA_GRAPH_SIZES'" >&2
        exit 2
    fi
    STARTUP_ARGS+=(--cudagraph-capture-sizes ${CUDA_GRAPH_SIZES//,/ })
elif [ "${STARTUP_MODE:-}" = "fast" ]; then
    # Batches past the largest captured size still run, without a graph. Capture beyond the
    # set costs boot time that a demo's traffic never uses.
    STARTUP_ARGS+=(--cudagraph-capture-sizes 1 2 4 8 16 32)
fi
case "${QUANT:-}" in
    ""|off) ;;
    kv_fp8) STARTUP_ARGS+=(--kv-cache-dtype fp8) ;;
    fp8) STARTUP_ARGS+=(--quantization fp8) ;;
    *) echo "QUANT must be off, fp8 or kv_fp8, not '$QUANT'" >&2; exit 2 ;;
esac
# A writable CACHE_DIR persists vLLM's compile cache and Triton's kernels across restarts when
# it is a mounted volume. Unwritable or unmounted, both stay ephemeral, which is today's
# behaviour (the platform Hugging Face Inference Endpoints runs on has no persistent storage).
CACHE_DIR=${CACHE_DIR:-/cache}
if mkdir -p "$CACHE_DIR" 2>/dev/null && [ -w "$CACHE_DIR" ]; then
    mkdir -p "$CACHE_DIR/vllm" "$CACHE_DIR/triton"
    export VLLM_CACHE_ROOT="$CACHE_DIR/vllm" TRITON_CACHE_DIR="$CACHE_DIR/triton"
else
    echo "CACHE_DIR '$CACHE_DIR' is not writable; compile and Triton caches stay ephemeral" >&2
fi
echo "startup: mode=${STARTUP_MODE:-default} quant=${QUANT:-off}" \
     "cuda_graph_sizes=${CUDA_GRAPH_SIZES:-vllm-default} cache=${VLLM_CACHE_ROOT:-ephemeral}"

# --dtype bfloat16: composed checkpoints carry torch_dtype null (granite-switch#126).
# flash_attn_version=2: avoids the FA3 schedule bug (granite-switch#139).
# --max-logprobs 256: lets chat clients that read labels by top-K ask for more than 20.
# --enable-prompt-tokens-details: reports prefix-cache hits in usage.input_tokens_details.
# No global temperature override: the plugin sends temperature 0 on every decision
# request, and chat clients keep the model's own default.
exec vllm serve "$MODEL_DIR" \
    --served-model-name "${SERVED_MODEL_NAME:-granite-switch}" \
    --host 0.0.0.0 --port "${PORT:-8000}" \
    --dtype bfloat16 \
    --tensor-parallel-size "$TP" \
    --max-model-len "${MAX_MODEL_LEN:-32768}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
    --attention-config.flash_attn_version=2 \
    --max-logprobs 256 \
    --enable-prompt-tokens-details \
    --enable-auto-tool-choice --tool-call-parser granite4 \
    "${SPEECH_ARGS[@]}" \
    "${STARTUP_ARGS[@]}" \
    "$@"
