# A vLLM OpenAI-compatible server for a composed granite-switch checkpoint, serving
# BOTH the standard /v1/chat/completions API and typesafe.ai's jev system-one API
# (/v1/systemone) on the same server -- one engine, one GPU KV cache, one container.
# The checkpoint's weights are NOT baked into this image (mounted separately at
# $MODEL_DIR / /repository, e.g. by HF Inference Endpoints' own deployment mechanism).
#
#   docker build --platform linux/amd64 -t <your-registry>/granite-switch-systemone:<tag> .
#
# Optional ASR support (build granite-switch[audio] in, and bake in a specific speech
# model's weights at a pinned revision -- see ASR_MODEL/ASR_REVISION below):
#   docker build --platform linux/amd64 --build-arg WITH_ASR=1 \
#       -t <your-registry>/granite-switch-systemone:<tag>-asr .
# vLLM 0.30, the version this plugin is tested on live (granite-switch 0.1.1 supports
# 0.26-0.30). The default tag is the CUDA 13 build (torch 2.13.0+cu130, the PyPI vLLM
# wheel) -- the same stack the live tests ran. The -cu129 tag ships torch 2.14.0,
# which the vLLM wheel's own torch==2.13.0 pin contradicts, so granite-switch[vllm]
# has no resolution against it.
FROM vllm/vllm-openai:v0.30.0
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG GRANITE_SWITCH_VERSION=0.1.1
ARG WITH_ASR=0

# Pinned to a specific, released granite-switch version (not `main`) so this image's
# behavior does not drift when upstream changes -- deliberately a tagged PyPI release,
# not a floating branch. transformers 5.17 is what granite-switch supports; [audio] is
# only installed when WITH_ASR=1 (it brings librosa and soundfile, used only by the
# ASR cascade). The base image's vLLM, torch, and torchcodec (CUDA 13 builds) are kept
# as they are: the constraint file pins them so nothing in granite-switch's dependency
# tree can move them.
RUN python3 -m pip list --format=freeze | grep -i -E "^(vllm|torch|torchaudio|torchcodec|torchvision|flashinfer-python)==" > /tmp/keep.txt \
    && if [ "$WITH_ASR" = "1" ]; then EXTRA="granite-switch[audio]"; else EXTRA="granite-switch[vllm]"; fi \
    && uv pip install --system --constraint /tmp/keep.txt \
        "${EXTRA}==${GRANITE_SWITCH_VERSION}" \
        "transformers==5.17.0" \
    && python3 -c "import granite_switch.vllm, vllm, transformers, torch; print(vllm.__version__, transformers.__version__, torch.__version__)"

# The ASR cascade's speech model, named by Hugging Face id in the checkpoint's config,
# baked in as a Hub cache at the revision this image was tested with, without the
# repo's unused .py files. The server runs with the Hub offline, so a missing/moved file
# fails startup instead of silently fetching something else.
# The pinned commit is downloaded by hash, not from `main`, so the build does not change when
# the model repository moves. A download by commit writes no refs/main, and the offline lookup
# for the revision "main" reads that file, so it is written here and then checked.
ARG ASR_MODEL=ibm-granite/granite-speech-5.0-470m-turboctc
ARG ASR_REVISION=947f59af40db9791170a0628cf0f3f4812d720f1
RUN if [ "$WITH_ASR" = "1" ]; then \
        hf download ${ASR_MODEL} --revision ${ASR_REVISION} --cache-dir /opt/hf-cache --exclude "*.py" > /dev/null \
        && mkdir -p /opt/hf-cache/models--${ASR_MODEL//\//--}/refs \
        && printf '%s' "${ASR_REVISION}" > /opt/hf-cache/models--${ASR_MODEL//\//--}/refs/main \
        && test -d /opt/hf-cache/models--${ASR_MODEL//\//--}/snapshots/${ASR_REVISION}; \
    fi
ENV HF_HUB_CACHE=/opt/hf-cache HF_HUB_OFFLINE=1

# The jev system-one API (/v1/systemone), a vllm.endpoint_plugins entry point mounted
# on this same server -- see README.md and src/granite_switch_systemone/plugin.py.
COPY pyproject.toml README.md /opt/granite-switch-systemone/
COPY src /opt/granite-switch-systemone/src
RUN uv pip install --system --no-deps /opt/granite-switch-systemone

COPY serve.sh /usr/local/bin/serve.sh
RUN chmod +x /usr/local/bin/serve.sh
# Endpoint plugins only load if VLLM_PLUGINS is set AND names them (stricter than
# other vLLM plugin groups, which default to "load everything" when VLLM_PLUGINS is
# unset): must include register_granite_switch too, or the Switch model class stops
# being registered the moment this allowlist is set at all.
ENV VLLM_PLUGINS=register_granite_switch,granite_switch_systemone \
    SYSTEMONE_ADAPTER_NAME=decision \
    SYSTEMONE_INVOCATION_MARKER="### DECISION"
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/serve.sh"]
