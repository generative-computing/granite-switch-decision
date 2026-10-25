#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Accuracy check for a live granite-switch-decision server, on public labelled data.

e2e_check.py shows the server speaks the wire formats; this shows the answers are still
good. It sends yes/no questions from the BoolQ validation split (google/boolq, CC BY-SA
3.0) to /v1/decisions as predicate questions and reports accuracy, log-loss and Brier
score. Rows are downloaded when the script runs, from Hugging Face's datasets server, and
are never stored in this repository. Standard library only.

  python3 scripts/accuracy_check.py --base-url http://localhost:8000

To compare two deployments (before and after an upgrade, or with a setting changed), save
the per-item results of each run and compare them on the same items:

  python3 scripts/accuracy_check.py --base-url URL_A --save before.json
  python3 scripts/accuracy_check.py --base-url URL_B --save after.json --compare before.json

The comparison reports items that only one run got right, and an exact McNemar p-value.
Differences in the third decimal place between runs are batching noise.

Exit code 0 means accuracy is at least --min-accuracy (default 0.75). That floor only
catches a grossly broken setup, for example a wrong invocation marker; judge regressions
with --compare. With the default 200 items the 95% interval on accuracy is about +/-5
points, so small differences need a paired comparison, not two headline numbers.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

ROWS_URL = "https://datasets-server.huggingface.co/rows"
PAGE = 100  # the datasets server returns at most 100 rows per request


def get_rows(url: str, attempts: int = 5) -> list[dict]:
    """The datasets server answers 5xx now and then; retry those, fail clearly on the rest."""
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                return json.load(resp)["rows"]
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == attempts:
                sys.exit(f"could not download BoolQ rows from {ROWS_URL}: HTTP {e.code}")
        except urllib.error.URLError as e:
            if attempt == attempts:
                sys.exit(f"could not download BoolQ rows from {ROWS_URL}: {e.reason}")
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def fetch_boolq(n: int) -> list[dict]:
    items: list[dict] = []
    while len(items) < n:
        query = urllib.parse.urlencode(
            {
                "dataset": "google/boolq",
                "config": "default",
                "split": "validation",
                "offset": len(items),
                "length": min(PAGE, n - len(items)),
            }
        )
        rows = get_rows(f"{ROWS_URL}?{query}")
        if not rows:
            break
        items += [r["row"] for r in rows]
    if len(items) < n:
        sys.exit(f"only {len(items)} BoolQ rows available, wanted {n}")
    return items


def ask(base_url: str, model: str, item: dict, timeout: float) -> float:
    """P(true) for one item, via /v1/decisions."""
    body = {
        "model": model,
        "input": item["passage"],
        "questions": [{"type": "predicate", "name": "q", "instructions": item["question"].strip().capitalize() + "?"}],
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/decisions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)["answers"][0]["probability"]


def wilson(correct: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def mcnemar_exact(only_a: int, only_b: int) -> float:
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="granite-switch")
    ap.add_argument("--n", type=int, default=200, help="BoolQ validation rows to use (default 200)")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--min-accuracy", type=float, default=0.75)
    ap.add_argument("--save", help="write per-item results to this JSON file")
    ap.add_argument("--compare", help="a file written by --save on an earlier run, to compare against")
    args = ap.parse_args()

    items = fetch_boolq(args.n)
    try:
        with ThreadPoolExecutor(args.concurrency) as pool:
            probs = list(pool.map(lambda it: ask(args.base_url, args.model, it, args.timeout), items))
    except urllib.error.URLError as e:
        sys.exit(f"cannot reach {args.base_url}: {e}")

    labels = [int(it["answer"]) for it in items]
    hits = [int((p >= 0.5) == bool(y)) for p, y in zip(probs, labels)]
    n = len(items)
    low, high = wilson(sum(hits), n)
    eps = 1e-12
    nll = -sum(math.log(max(p if y else 1 - p, eps)) for p, y in zip(probs, labels)) / n
    brier = sum((p - y) ** 2 for p, y in zip(probs, labels)) / n
    print(f"BoolQ validation, first {n} rows, {args.base_url}")
    print(f"  accuracy {sum(hits) / n:.3f}  (95% interval {low:.3f} to {high:.3f})")
    print(f"  log-loss {nll:.3f}   Brier {brier:.3f}")

    if args.save:
        with open(args.save, "w") as f:
            json.dump({"n": n, "probs": probs, "labels": labels}, f)
        print(f"  saved {args.save}")

    if args.compare:
        with open(args.compare) as f:
            before = json.load(f)
        if before["labels"] != labels:
            sys.exit(f"{args.compare} was made on different items; rerun both with the same --n")
        before_hits = [int((p >= 0.5) == bool(y)) for p, y in zip(before["probs"], labels)]
        only_before = sum(1 for a, b in zip(before_hits, hits) if a and not b)
        only_now = sum(1 for a, b in zip(before_hits, hits) if b and not a)
        drift = max(abs(a - b) for a, b in zip(before["probs"], probs))
        print(
            f"  against {args.compare}: right only before {only_before}, right only now {only_now}, "
            f"McNemar exact p={mcnemar_exact(only_before, only_now):.2f}, largest probability change {drift:.3f}"
        )

    if sum(hits) / n < args.min_accuracy:
        print(f"FAIL: accuracy below --min-accuracy {args.min_accuracy}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
