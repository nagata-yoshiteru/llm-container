#!/usr/bin/env python3
"""bench-ttft.py — TTFT / throughput benchmark for DSpark vLLM API.

Usage:
    python3 scripts/bench-ttft.py [--prompt-len 256,512,1024,2048,4096,65536] [--num-prompts 20] [--output results/bench-<ts>.json]

Tests each prompt length independently. Reports TTFT mean/P50/P99 and TPOT.
"""

import argparse, json, os, statistics, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import Request, urlopen
from urllib.error import URLError

API = os.environ.get("VLLM_API_URL", "http://127.0.0.1:8888")
MODEL = os.environ.get("VLLM_MODEL", "deepseek-v4-flash-0731")


def make_prompt(target_tokens: int) -> str:
    """Generate a prompt of approximately target_tokens."""
    # ~1 token per 4 chars for English-ish text
    return "hello " * (target_tokens * 4 // 6)


def single_request(prompt: str, max_tokens: int = 32) -> dict:
    """Send one completion request, return timing + usage info."""
    payload = json.dumps({
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }).encode()

    req = Request(
        f"{API}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
        t1 = time.perf_counter()

        usage = body.get("usage", {})
        text = body.get("choices", [{}])[0].get("text", "")
        return {
            "ok": True,
            "latency_ms": (t1 - t0) * 1000,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "text_len": len(text),
        }
    except Exception as e:
        t1 = time.perf_counter()
        return {"ok": False, "latency_ms": (t1 - t0) * 1000, "error": str(e)}


def bench_length(prompt_len: int, num_prompts: int) -> dict:
    """Run num_prompts requests at a given prompt length."""
    prompt = make_prompt(prompt_len)
    results = []

    # Sequential to avoid overwhelming the server with large prompts
    for i in range(num_prompts):
        r = single_request(prompt)
        r["prompt_len_target"] = prompt_len
        r["request_index"] = i
        results.append(r)
        if not r["ok"]:
            print(f"    [{i+1}/{num_prompts}] FAIL: {r.get('error','')[:60]}", file=sys.stderr)

    ok = [r for r in results if r["ok"]]
    if not ok:
        return {"prompt_len": prompt_len, "num_ok": 0, "num_fail": len(results)}

    latencies = [r["latency_ms"] for r in ok]
    prompt_tokens = [r["prompt_tokens"] for r in ok]
    completion_tokens = [r["completion_tokens"] for r in ok]

    return {
        "prompt_len": prompt_len,
        "actual_prompt_tokens_mean": statistics.mean(prompt_tokens),
        "num_ok": len(ok),
        "num_fail": len(results) - len(ok),
        "latency_mean_ms": statistics.mean(latencies),
        "latency_p50_ms": statistics.median(latencies),
        "latency_p99_ms": sorted(latencies)[int(len(latencies) * 0.99)],
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
        "latency_stdev_ms": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "completion_tokens_mean": statistics.mean(completion_tokens),
        "throughput_req_s": len(ok) / (sum(latencies) / 1000) if sum(latencies) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-len", default="256,512,1024,2048,4096,65536",
                        help="Comma-separated prompt lengths to test")
    parser.add_argument("--num-prompts", type=int, default=10,
                        help="Requests per prompt length")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/bench-<ts>.json)")
    args = parser.parse_args()

    lengths = [int(x) for x in args.prompt_len.split(",")]
    os.makedirs("results", exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = args.output or f"results/bench-{ts}.json"

    print(f"vLLM TTFT benchmark — {API} model={MODEL}")
    print(f"Prompt lengths: {lengths}")
    print(f"Requests per length: {args.num_prompts}")
    print()

    all_results = []
    for pl in lengths:
        print(f"  benchmarking prompt_len={pl} ...", end="", flush=True)
        r = bench_length(pl, args.num_prompts)
        all_results.append(r)
        if r["num_ok"] > 0:
            print(f"  {r['num_ok']}/{r['num_ok']+r['num_fail']} ok  "
                  f"latency={r['latency_mean_ms']:.0f}±{r['latency_stdev_ms']:.0f}ms  "
                  f"p50={r['latency_p50_ms']:.0f}ms  p99={r['latency_p99_ms']:.0f}ms  "
                  f"tokens={r['actual_prompt_tokens_mean']:.0f}")
        else:
            print(f"  ALL FAILED")

    # Summary table
    print()
    print("=" * 90)
    print(f"{'prompt_len':>12} {'tokens':>8} {'ok':>4} {'mean_ms':>10} {'p50_ms':>10} {'p99_ms':>10} {'stdev':>8}")
    print("-" * 90)
    for r in all_results:
        if r["num_ok"] > 0:
            print(f"{r['prompt_len']:>12} {r['actual_prompt_tokens_mean']:>8.0f} "
                  f"{r['num_ok']:>4} {r['latency_mean_ms']:>10.0f} {r['latency_p50_ms']:>10.0f} "
                  f"{r['latency_p99_ms']:>10.0f} {r['latency_stdev_ms']:>8.0f}")
        else:
            print(f"{r['prompt_len']:>12} {'—':>8} {'0':>4} {'FAIL':>10}")
    print("=" * 90)

    output = {
        "timestamp": ts,
        "api": API,
        "model": MODEL,
        "num_prompts": args.num_prompts,
        "results": all_results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
