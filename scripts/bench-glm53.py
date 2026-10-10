#!/usr/bin/env python3
"""Fixed-length, streaming GLM speed probe (not an answer-quality test).

Run the same command after each server restart, changing only --label/output.
Counts server usage tokens, including reasoning; SSE chunks are NOT tokens.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import time
from urllib.request import Request, urlopen

PROMPTS = {
    "prose": "Explain how a relational database executes a SQL query, from parsing "
             "through optimization to execution. Give concrete examples.",
    "code": "Write a Python LRU cache with O(1) get and put operations. Explain "
            "the invariants and cover capacity zero and replacement of existing keys.",
}


def request(args, kind):
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": PROMPTS[kind]}],
        "temperature": 0, "seed": 42,
        "max_tokens": args.tokens, "min_tokens": args.tokens, "ignore_eos": True,
        "chat_template_kwargs": {"reasoning_effort": "max"},
        "stream": True, "stream_options": {"include_usage": True},
    }
    req = Request(args.url.rstrip("/") + "/v1/chat/completions",
                  json.dumps(payload).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    first = None
    usage = None
    text = []
    done = False
    with urlopen(req, timeout=600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                delta = choice.get("delta", {})
                fragment = (delta.get("reasoning") or delta.get("reasoning_content") or "") + (delta.get("content") or "")
                if fragment:
                    first = first if first is not None else time.perf_counter()
                    text.append(fragment)
    elapsed = time.perf_counter() - start
    if not done or first is None or not usage or usage["completion_tokens"] != args.tokens:
        raise RuntimeError(f"Incomplete fixed-length sample: done={done}, usage={usage}")
    return {"kind": kind, "usage": usage, "ttft_s": first - start,
            "elapsed_s": elapsed, "e2e_tok_s": usage["completion_tokens"] / elapsed,
            "output_sha256": hashlib.sha256("".join(text).encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8910")
    parser.add_argument("--model", default="glm-5.3-flash-uncensored")
    parser.add_argument("--label", required=True, help="Manually recorded running configuration")
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens <= 0 or args.repeats <= 0:
        parser.error("tokens and repeats must be positive")
    if args.output.exists():
        parser.error("output exists; use a new filename")
    result = {"timestamp": datetime.now(timezone.utc).isoformat(),
              "label": args.label, "url": args.url, "model": args.model,
              "tokens": args.tokens, "repeats": args.repeats, "prompts": PROMPTS,
              "sampling": {"temperature": 0, "seed": 42, "reasoning_effort": "max",
                           "ignore_eos": True, "min_tokens": args.tokens},
              "note": "Fixed-length throughput only; prefix reuse allowed. Warmup excluded. "
                      "Requires an otherwise idle server. Label is not server-verified.",
              "samples": [], "summary": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        # Warm every DP rank before measured requests (three internal DP ranks).
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(lambda _: request(args, "prose"), range(3)))
        for concurrency in (1, 2):
            for kind in PROMPTS:
                rates = []
                for repeat in range(args.repeats):
                    start = time.perf_counter()
                    with ThreadPoolExecutor(max_workers=concurrency) as pool:
                        rows = list(pool.map(lambda _: request(args, kind), range(concurrency)))
                    wall = time.perf_counter() - start
                    rates.extend(row["e2e_tok_s"] for row in rows)
                    result["samples"].append({"concurrency": concurrency, "kind": kind,
                                              "repeat": repeat, "wall_s": wall, "rows": rows})
                    print(f"C{concurrency} {kind} #{repeat + 1}: " +
                          ", ".join(f"{row['e2e_tok_s']:.2f} tok/s" for row in rows), flush=True)
                result["summary"].append({"concurrency": concurrency, "kind": kind,
                                          "median_per_request_tok_s": statistics.median(rates)})
    except Exception as exc:
        result["error"] = str(exc)
        raise
    finally:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
