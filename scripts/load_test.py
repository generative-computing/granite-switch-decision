#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Throughput and latency of /v1/decisions on a live granite-switch-systemone server.

Standard library only. Each client loops on one keep-alive connection and sends the same kind of
request: three questions (a predicate, a 20-way choice and a 5-level score) over a state of about
230 tokens. Per level it prints requests per second, p50/p95/p99 latency, the HTTP statuses seen,
and the server's prefix-cache hit rate (read from vLLM's /metrics when it is exposed).

  python3 scripts/load_test.py --base-url http://localhost:8000
  python3 scripts/load_test.py --base-url http://localhost:8000 --mode identical
  python3 scripts/load_test.py --base-url http://localhost:8000 --levels 1,8,32 --seconds 30

--mode unique (the default) puts a fresh identifier at the start of every state, so requests share
only the questions' prefix and the prefix cache cannot serve them whole. --mode identical sends the
same body every time: the cache serves almost every token, which flatters throughput. Run both and
report both; the identical figure is an upper bound, not what a mixed workload will see.

Run it from a different machine than the server, on a server with no other traffic, and give each
level enough time to settle (the default is 45 seconds). One run per level gives no spread.

A caller that gets HTTP 503 (see SYSTEMONE_MAX_CONCURRENT_REQUESTS) here resends immediately, which
is harsher than a client that backs off, so a capped server's throughput looks worse than a
well-behaved caller would see. Use --backoff SECONDS to wait after a non-200.
"""

from __future__ import annotations

import argparse
import collections
import http.client
import json
import re
import statistics
import sys
import threading
import time
import urllib.parse
import uuid

TICKET = "A customer writes: I was charged twice for my subscription this month and want the duplicate refunded."
QUESTIONS = [
    {"type": "predicate", "instructions": "Is a refund requested?"},
    {"type": "choice", "instructions": "Which team?", "choices": [{"value": f"team {j}"} for j in range(20)]},
    {"type": "score", "instructions": "How formal is this?", "levels": [{"label": str(i)} for i in range(5)]},
]
HEADERS = {"Content-Type": "application/json"}


def make_body(model: str, mode: str) -> str:
    state = TICKET * 4
    if mode == "unique":
        state = f"Ticket {uuid.uuid4().hex}. {state}"
    return json.dumps({"model": model, "input": state, "questions": QUESTIONS})


def percentile(sorted_values: list[float], q: float) -> float:
    return (
        sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))] * 1000
        if sorted_values
        else float("nan")
    )


class Target:
    def __init__(self, base_url: str, timeout: float) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            sys.exit(f"--base-url must look like http://host:port, got {base_url!r}")
        self.https = parsed.scheme == "https"
        self.host = parsed.hostname
        self.port = parsed.port or (443 if self.https else 80)
        self.timeout = timeout

    def connect(self) -> http.client.HTTPConnection:
        cls = http.client.HTTPSConnection if self.https else http.client.HTTPConnection
        return cls(self.host, self.port, timeout=self.timeout)

    def get(self, path: str) -> tuple[int, bytes]:
        conn = self.connect()
        try:
            conn.request("GET", path)
            resp = conn.getresponse()
            return resp.status, resp.read()
        finally:
            conn.close()


def cache_counters(target: Target) -> tuple[float, float] | None:
    """(hits, queries) summed over vLLM's prefix-cache counters, or None when /metrics lacks them."""
    try:
        status, raw = target.get("/metrics")
    except OSError:
        return None
    if status != 200:
        return None
    text = raw.decode(errors="replace")

    def total(name: str) -> float | None:
        values = [
            float(m.group(1)) for m in re.finditer(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.e+-]+)$", text, re.M)
        ]
        return sum(values) if values else None

    hits, queries = total("vllm:prefix_cache_hits_total"), total("vllm:prefix_cache_queries_total")
    return (hits, queries) if hits is not None and queries is not None else None


def run_level(target: Target, model: str, mode: str, clients: int, seconds: float, backoff: float) -> None:
    stop = time.time() + seconds
    results: list[tuple[int, float, str | None]] = []
    lock = threading.Lock()

    def loop() -> None:
        conn = target.connect()
        while time.time() < stop:
            started = time.time()
            error = None
            try:
                conn.request("POST", "/v1/decisions", body=make_body(model, mode), headers=HEADERS)
                resp = conn.getresponse()
                resp.read()
                status = resp.status
            except OSError as e:
                status, error = 0, f"{type(e).__name__}: {e}"
                conn.close()
                conn = target.connect()
            elapsed = time.time() - started
            with lock:
                results.append((status, elapsed, error))
            if status != 200 and backoff:
                time.sleep(backoff)
        conn.close()

    before = cache_counters(target)
    threads = [threading.Thread(target=loop) for _ in range(clients)]
    started = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - started
    after = cache_counters(target)

    ok = sorted(elapsed for status, elapsed, _ in results if status == 200)
    hit = ""
    if before and after and after[1] > before[1]:
        hit = f"  prefix-cache hit {100 * (after[0] - before[0]) / (after[1] - before[1]):.1f}%"
    statuses = dict(collections.Counter(status for status, _, _ in results))
    print(
        f"{mode:9s} {clients:4d} clients: {len(ok) / wall:7.1f} req/s  p50 {percentile(ok, 0.5):6.0f} ms  "
        f"p95 {percentile(ok, 0.95):6.0f} ms  p99 {percentile(ok, 0.99):6.0f} ms  statuses {statuses}{hit}",
        flush=True,
    )
    for message, n in collections.Counter(error for _, _, error in results if error).most_common(3):
        print(f"      {n} x {message}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True, help="server root, for example http://localhost:8000")
    ap.add_argument("--model", default="granite-switch", help="the served model name")
    ap.add_argument("--mode", choices=("unique", "identical"), default="unique")
    ap.add_argument("--levels", default="1,8,32,64,128", help="comma-separated client counts")
    ap.add_argument("--seconds", type=float, default=45.0, help="seconds per level")
    ap.add_argument(
        "--backoff", type=float, default=0.0, help="seconds to wait after a non-200 (default: resend at once)"
    )
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args()
    try:
        levels = [int(x) for x in args.levels.split(",") if x.strip()]
    except ValueError:
        ap.error("--levels must be comma-separated integers")
    if not levels or min(levels) < 1:
        ap.error("--levels needs at least one client count of 1 or more")

    target = Target(args.base_url, args.timeout)
    try:
        samples = []
        for _ in range(11):
            started = time.time()
            status, _ = target.get("/v1/models")
            samples.append(time.time() - started)
            if status != 200:
                sys.exit(f"GET /v1/models returned {status}: is {args.base_url} the server?")
    except OSError as e:
        sys.exit(f"cannot reach {args.base_url}: {e}")
    samples = sorted(samples[1:])
    median_ms = statistics.median(samples) * 1000
    print(
        f"server {args.base_url}, mode {args.mode}, {args.seconds:g} s per level; "
        f"round trip (new connection each) min {samples[0] * 1000:.1f} ms, median {median_ms:.1f} ms",
        flush=True,
    )
    if cache_counters(target) is None:
        print("(the server's /metrics has no prefix-cache counters; the hit rate is not shown)", flush=True)
    for clients in levels:
        run_level(target, args.model, args.mode, clients, args.seconds, args.backoff)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
