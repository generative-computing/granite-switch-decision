#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Long-context checks for a live granite-switch-systemone server started with the
8b checkpoint and --max-model-len 131072 (the checkpoint's maximum). Needs the
checkpoint's tokenizer (transformers, unlike the other scripts, which are stdlib-only)
to size the probe states. The 262,144-token danger-zone cap clamps to --max-model-len,
so this exercises it up to 131,072; no released Switch checkpoint has more context.

  python3 scripts/longctx_check.py --base-url http://localhost:8000 \
      --tokenizer /path/to/granite-switch-4.1-8b

Exit code 0 means every check passed. Checks:

  default_caps      a ~98k-token state is rejected without the header (the 32k
                    per-question cap fires first; the 64k per-request cap is
                    checked separately with three ~22k questions)
  danger_zone       the same state with TypeSafe-Danger-Zone-Long-Context: true is
                    answered in one engine read, and its usage counts the tokens
  max_model_len     a ~152k-token state with the header is rejected at the
                    131,072 backstop
  decisions_route   /v1/decisions has no danger-zone escape: the big state is
                    rejected there too
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

FILLER = (
    "The quick brown fox jumps over the lazy dog while the engine keeps a running "
    "account of every payout that failed, every ticket that was routed, and every "
    "decision the queue made about them. "
)


def build_state(tokenizer, target_tokens: int) -> tuple[str, int]:
    """A deterministic text of at least target_tokens tokens. A few tokenizer passes,
    not one per append; concatenation merges tokens at chunk boundaries, so converge
    on the target with a conservative per-chunk estimate."""
    n_chunk = len(tokenizer(FILLER, add_special_tokens=False)["input_ids"])
    per = max(n_chunk - 4, 1)
    k = max(1, target_tokens // per + 2)
    while True:
        text = FILLER * k
        n = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if n >= target_tokens:
            return text, n
        k += (target_tokens - n) // per + 1


def post(url, body, headers=None, timeout=1800):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--tokenizer", required=True, help="path to the granite-switch-4.1-8b snapshot")
    ap.add_argument("--model", default="granite-switch")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    state_90k, _ = build_state(tok, 90_000)
    state_140k, _ = build_state(tok, 140_000)
    state_5k, _ = build_state(tok, 5_000)
    state_22k, _ = build_state(tok, 22_000)
    instr, _ = build_state(tok, 28_500)
    results = []

    def check(name, ok, detail=""):
        results.append(ok)
        print(f"{'PASS' if ok else 'FAIL'} {name} {detail}")

    q = {"q": {"type": "noul", "instructions": "Does this need urgent attention?"}}
    body = {"model": args.model, "state": state_90k, "questions": q}
    hdr = {"TypeSafe-Danger-Zone-Long-Context": "true"}

    # Without the header the default caps reject: the 32k per-question cap fires first
    # on a one-question request this big.
    code, resp = post(f"{args.base_url}/v1/systemone", body)
    check(
        "90k state, no header -> 400 at the default cap",
        code == 400 and ("32000" in str(resp) or "64000" in str(resp)),
        f"code={code}",
    )

    # With the header the caps rise to min(262144, max_model_len): answered in one read.
    t0 = time.time()
    code, resp = post(f"{args.base_url}/v1/systemone", body, hdr)
    dt = time.time() - t0
    ok = code == 200 and 0.0 <= resp["answers"]["q"]["noul"] <= 1.0
    usage = resp.get("usage", {}) if code == 200 else {}
    check(
        "90k state, header -> 200 answered",
        ok,
        f"code={code} p={resp['answers']['q'] if code == 200 else str(resp)[:120]} {dt:.1f}s",
    )
    if ok:
        check(
            "90k usage counts the tokens",
            usage.get("input_tokens", 0) > 80_000,
            f"input_tokens={usage.get('input_tokens')}",
        )

    # Beyond max-model-len even with the header: the 131,072 backstop.
    body3 = {"model": args.model, "state": state_140k, "questions": q}
    code, resp = post(f"{args.base_url}/v1/systemone", body3, hdr)
    check("140k state, header -> 400 at 131072 backstop", code == 400 and "131072" in str(resp), f"code={code}")

    # /v1/decisions has no danger-zone concept.
    dbody = {
        "model": args.model,
        "input": state_90k,
        "questions": [{"type": "predicate", "instructions": "Does this need urgent attention?", "name": "q"}],
    }
    code, resp = post(f"{args.base_url}/v1/decisions", dbody)
    check(
        "90k input via /v1/decisions -> 400 at the default cap",
        code == 400 and ("32000" in str(resp) or "64000" in str(resp)),
        f"code={code}",
    )

    # The 64k per-REQUEST cap: three questions each under 32k, together over 64k.
    body4 = {
        "model": args.model,
        "state": state_22k,
        "questions": {
            f"q{i}": {"type": "noul", "instructions": f"Question number {i}: does this need attention?"}
            for i in range(3)
        },
    }
    code, resp = post(f"{args.base_url}/v1/systemone", body4)
    check(
        "3x22k questions, no header -> 400 at 64k per-request cap",
        code == 400 and "64000" in str(resp),
        f"code={code} {str(resp)[:150]}",
    )
    code, resp = post(f"{args.base_url}/v1/systemone", body4, hdr)
    ok = code == 200 and all(0.0 <= a["noul"] <= 1.0 for a in resp["answers"].values())
    check("3x22k questions, header -> 200 answered", ok, f"code={code}")

    # A single question just over the 32k per-question cap, without and with the header.
    body5 = {"model": args.model, "state": state_5k, "questions": {"q": {"type": "noul", "instructions": instr}}}
    code, resp = post(f"{args.base_url}/v1/systemone", body5)
    check(
        "~33k question, no header -> 400 at 32k per-question cap", code == 400 and "32000" in str(resp), f"code={code}"
    )
    t0 = time.time()
    code, resp = post(f"{args.base_url}/v1/systemone", body5, hdr)
    dt = time.time() - t0
    ok = code == 200 and 0.0 <= resp["answers"]["q"]["noul"] <= 1.0
    check("~33k question, header -> 200 answered", ok, f"code={code} {dt:.1f}s")

    print(f"{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
