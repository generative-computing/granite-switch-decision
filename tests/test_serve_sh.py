# SPDX-License-Identifier: Apache-2.0
"""No-GPU checks for serve.sh's environment switches, with a stub `vllm` that records its
arguments and the cache environment it would run with.

python3 -m pytest tests/test_serve_sh.py -v --tb=short
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SERVE_SH = Path(__file__).resolve().parent.parent / "serve.sh"
OVERRIDE = '{"asr_enabled": false}'


def _run(tmp_path: Path, extra_env: dict[str, str], *args: str, asr_enabled: bool = True):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "vllm"
    stub.write_text(
        "#!/bin/sh\n"
        'printf "%s\\n" "$@" > "$STUB_OUT"\n'
        'env | grep -E "^(VLLM_CACHE_ROOT|TRITON_CACHE_DIR)=" > "$STUB_ENV" || true\n'
    )
    stub.chmod(0o755)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps({"asr_enabled": asr_enabled}))
    out = tmp_path / "args.txt"
    env_out = tmp_path / "env.txt"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}",
        "MODEL_DIR": str(model_dir),
        "TENSOR_PARALLEL_SIZE": "1",
        "STUB_OUT": str(out),
        "STUB_ENV": str(env_out),
        "CACHE_DIR": str(tmp_path / "cache"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_CACHE": str(tmp_path / "empty-cache"),
        # these tests are about the speech model, not the audio libraries
        "DECISION_PREFLIGHT_AUDIO_MODULES": "",
        **extra_env,
    }
    done = subprocess.run(["bash", str(SERVE_SH), *args], env=env, capture_output=True, text=True)
    argv = out.read_text().splitlines() if out.exists() else []
    cache_env = dict(line.split("=", 1) for line in env_out.read_text().splitlines()) if env_out.exists() else {}
    return done, argv, cache_env


def test_speech_input_is_off_by_default_and_needs_no_speech_model(tmp_path):
    done, argv, _ = _run(tmp_path, {})
    assert done.returncode == 0, done.stderr
    assert argv[argv.index("--hf-overrides") + 1] == OVERRIDE


def test_speech_input_on_without_the_model_stops_before_vllm(tmp_path):
    done, argv, _ = _run(tmp_path, {"DECISION_SPEECH_INPUT": "1"})
    assert done.returncode == 1
    assert argv == []  # vllm never started
    assert "WITH_ASR=1" in done.stderr


def test_speech_input_on_with_the_model_cached_starts_without_the_override(tmp_path):
    speech = "models--ibm-granite--granite-speech-5.0-470m-turboctc"
    cache = tmp_path / "empty-cache" / speech / "snapshots" / "abc"
    cache.mkdir(parents=True)
    (cache / "config.json").write_text("{}")
    done, argv, _ = _run(tmp_path, {"DECISION_SPEECH_INPUT": "1"})
    assert done.returncode == 0, done.stderr
    assert "--hf-overrides" not in argv


def test_speech_input_on_without_the_audio_libraries_stops_before_vllm(tmp_path):
    done, argv, _ = _run(
        tmp_path,
        {
            "DECISION_SPEECH_INPUT": "1",
            "DECISION_PREFLIGHT_AUDIO_MODULES": "no_such_audio_module_xyz",
        },
    )
    assert done.returncode == 1
    assert argv == []
    assert "no_such_audio_module_xyz" in done.stderr
    assert "HTTP 500" in done.stderr


def test_extra_arguments_still_reach_vllm(tmp_path):
    done, argv, _ = _run(tmp_path, {}, "--api-key", "k")
    assert done.returncode == 0, done.stderr
    assert argv[-2:] == ["--api-key", "k"]


@pytest.mark.parametrize("value", ["2", "yes"])
def test_bad_switch_value_is_refused(tmp_path, value):
    done, argv, _ = _run(tmp_path, {"DECISION_SPEECH_INPUT": value})
    assert done.returncode == 2
    assert argv == []


def test_empty_switch_value_falls_back_to_off(tmp_path):
    done, argv, _ = _run(tmp_path, {"DECISION_SPEECH_INPUT": ""})
    assert done.returncode == 0


# Startup tuning: every option defaults to today's flags, and a set-but-unrecognised value
# stops the server before vllm starts.


def test_startup_defaults_add_no_flags_and_log_the_banner(tmp_path):
    done, argv, _ = _run(tmp_path, {})
    assert done.returncode == 0, done.stderr
    assert "--optimization-level" not in argv
    assert "--enforce-eager" not in argv
    assert "--cudagraph-capture-sizes" not in argv
    assert "--kv-cache-dtype" not in argv
    assert "--quantization" not in argv
    assert "startup: mode=default" in done.stdout


def test_startup_mode_fast_compiles_less_and_trims_capture_sizes(tmp_path):
    done, argv, _ = _run(tmp_path, {"STARTUP_MODE": "fast"})
    assert done.returncode == 0, done.stderr
    assert argv[argv.index("--optimization-level") + 1] == "1"
    i = argv.index("--cudagraph-capture-sizes")
    assert argv[i + 1 : i + 7] == ["1", "2", "4", "8", "16", "32"]


def test_startup_mode_eager_skips_compilation_and_graphs(tmp_path):
    done, argv, _ = _run(tmp_path, {"STARTUP_MODE": "eager"})
    assert done.returncode == 0, done.stderr
    assert "--enforce-eager" in argv
    assert "--optimization-level" not in argv


def test_startup_mode_fast_respects_an_explicit_capture_size_set(tmp_path):
    done, argv, _ = _run(tmp_path, {"STARTUP_MODE": "fast", "CUDA_GRAPH_SIZES": "4,8"})
    assert done.returncode == 0, done.stderr
    i = argv.index("--cudagraph-capture-sizes")
    assert argv[i + 1 : i + 3] == ["4", "8"]


def test_cuda_graph_sizes_alone_trims_capture_without_touching_optimization(tmp_path):
    done, argv, _ = _run(tmp_path, {"CUDA_GRAPH_SIZES": "1,2,4"})
    assert done.returncode == 0, done.stderr
    i = argv.index("--cudagraph-capture-sizes")
    assert argv[i + 1 : i + 4] == ["1", "2", "4"]
    assert "--optimization-level" not in argv


@pytest.mark.parametrize("value", ["turbo", "FAST"])
def test_bad_startup_mode_is_refused(tmp_path, value):
    done, argv, _ = _run(tmp_path, {"STARTUP_MODE": value})
    assert done.returncode == 2
    assert argv == []
    assert value in done.stderr


@pytest.mark.parametrize("value", ["1,x", "1;2", ",", "1,,2"])
def test_bad_cuda_graph_sizes_are_refused(tmp_path, value):
    done, argv, _ = _run(tmp_path, {"CUDA_GRAPH_SIZES": value})
    assert done.returncode == 2
    assert argv == []
    assert value in done.stderr


# Both quantization values were measured failing inside vLLM against the 8b checkpoint's aLoRA
# adapters, so any value other than unset/off is refused at startup rather than failing later.


@pytest.mark.parametrize("value", ["kv_fp8", "fp8", "int4"])
def test_any_quant_value_is_refused_with_the_reason(tmp_path, value):
    done, argv, _ = _run(tmp_path, {"QUANT": value})
    assert done.returncode == 2
    assert argv == []
    assert "not supported for granite-switch" in done.stderr
    assert "aLoRA" in done.stderr


@pytest.mark.parametrize("value", ["", "off"])
def test_quant_empty_and_off_are_accepted(tmp_path, value):
    done, argv, _ = _run(tmp_path, {"QUANT": value})
    assert done.returncode == 0, done.stderr
    assert "--quantization" not in argv
    assert "--kv-cache-dtype" not in argv


def test_writable_cache_dir_persists_compile_and_triton_caches(tmp_path):
    done, _, cache_env = _run(tmp_path, {})
    assert done.returncode == 0, done.stderr
    assert cache_env["VLLM_CACHE_ROOT"] == str(tmp_path / "cache" / "vllm")
    assert cache_env["TRITON_CACHE_DIR"] == str(tmp_path / "cache" / "triton")


def test_unwritable_cache_dir_falls_back_to_ephemeral_caches(tmp_path):
    # A file where a directory is needed: mkdir fails for any user, root included.
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    done, argv, cache_env = _run(tmp_path, {"CACHE_DIR": str(blocked)})
    assert done.returncode == 0, done.stderr
    assert argv  # vllm still started
    assert cache_env == {}
    assert "not writable" in done.stderr
    assert "startup: mode=default" in done.stdout
