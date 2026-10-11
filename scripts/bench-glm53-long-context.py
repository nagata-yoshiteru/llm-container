#!/usr/bin/env python3
"""Test independent near-limit contexts, retaining them through concurrent decode.

Uses the existing streaming/metrics harness. Fixed-length output keeps earlier
requests alive while later requests prefill; measured decode overlap verifies
that all contexts were resident together. Never restarts or reconfigures servers.
"""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import re
import threading
import time
import urllib.request
import uuid


def first_codes(text):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start():])
        except ValueError:
            continue
        if isinstance(obj, dict) and set(obj) == {"RED", "GREEN", "BLUE"}:
            return obj
    return None


def make_prompt(count, codes):
    lines = [f"Record {i:06d}: warehouse {(i * 17) % 997}, inventory {(i * 7919) % 100003}, "
             f"quantity {(i * 13) % 211}. Confirm delivery before updating the inventory ledger."
             for i in range(count)]
    for fraction, (label, value) in zip((0.1, 0.5, 0.9), codes.items()):
        lines[int(count * fraction)] += f" AUDIT NOTE: the exact verification code for {label} is {value}."
    return ("Read these reference records and remember the three audit verification codes.\n\n" +
            "\n".join(lines) + "\n\nFirst return a JSON object mapping RED, GREEN, and BLUE to their exact "
            "verification codes from the records. Then explain how the records support your answer.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8910")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-tokens", type=int, default=1040384)
    parser.add_argument("--output-tokens", type=int, default=2048)
    parser.add_argument("--concurrencies", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--memory-hosts", nargs="*", default=["192.168.4.1", "192.168.4.2", "192.168.4.3"])
    args = parser.parse_args()
    if args.input_tokens < 4096 or args.output_tokens < 1 or any(c < 1 or c > 8 for c in args.concurrencies):
        parser.error("Use input >=4096, positive output, and concurrency 1..8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / "long-context-start.json"
    if marker.exists():
        parser.error("Output directory already contains a long-context run")
    spec = importlib.util.spec_from_file_location("glm53_eval", Path(__file__).with_name("bench-glm53-eval.py"))
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    bench.API, bench.OUT = args.url.rstrip("/"), args.output_dir
    models = json.load(urllib.request.urlopen(bench.API + "/v1/models", timeout=10))["data"]
    model = next(m for m in models if m["id"] == bench.MODEL)
    limit = model.get("max_model_len")
    if not limit or args.input_tokens + 512 + args.output_tokens > limit:
        parser.error(f"Requested input+output budget exceeds reported context limit {limit}")
    marker.write_text(json.dumps({"at_utc": datetime.now(timezone.utc).isoformat(),
        "input_target": args.input_tokens, "input_tolerance": 512, "output_tokens": args.output_tokens,
        "concurrencies": args.concurrencies, "server_model": model,
        "notes": ["Unique audit codes and cache_salt per request; no prefix-cache sharing.",
                  "Forced-length output is a capacity test, not a natural answer-length benchmark.",
                  "Decode overlap must be positive to demonstrate simultaneous full contexts."]}, indent=2))
    grades = []
    for concurrency in args.concurrencies:
        jobs, expected = [], {}
        label = f"long-n{args.input_tokens}-c{concurrency}"
        for i in range(concurrency):
            name = f"{label}-{i}"
            codes = {key: f"{key}-{uuid.uuid4().hex[:12]}" for key in ("RED", "GREEN", "BLUE")}
            count = max(100, int(args.input_tokens / 37))
            for _ in range(8):
                prompt = make_prompt(count, codes)
                actual = bench.post("/tokenize", {"model": bench.MODEL,
                    "messages": [{"role": "user", "content": prompt}],
                    "chat_template_kwargs": {"reasoning_effort": "max"}})["count"]
                if abs(actual - args.input_tokens) <= 512:
                    break
                count = max(100, round(count * args.input_tokens / actual))
            if abs(actual - args.input_tokens) > 512 or actual + args.output_tokens > limit:
                raise RuntimeError(f"Could not size {name}: {actual}")
            (bench.OUT / (name + "-input.json")).write_text(json.dumps({
                "tokens": actual, "expected": codes, "prompt": prompt}))
            expected[name] = {"tokens": actual, "codes": codes}
            jobs.append((name, {"messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.output_tokens, "min_tokens": args.output_tokens, "ignore_eos": True,
                "temperature": 0, "seed": 42, "cache_salt": uuid.uuid4().hex,
                "chat_template_kwargs": {"reasoning_effort": "max"}}))
        stop = threading.Event()

        def monitor():
            with (bench.OUT / (label + "-monitor.jsonl")).open("w") as log:
                while not stop.is_set():
                    sample = {"at_unix": time.time(), "hosts": {}}
                    # This harness runs on head; cumulative counters distinguish
                    # old swap occupancy from paging caused by this test.
                    try:
                        counters = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
                        sample["client_host_swap_pages"] = {k: int(counters[k]) for k in ("pswpin", "pswpout")}
                    except (OSError, KeyError, ValueError) as exc:
                        sample["swap_counter_error"] = repr(exc)
                    try:
                        text = urllib.request.urlopen(bench.API + "/metrics", timeout=3).read().decode()
                        sample["metrics"] = [line for line in text.splitlines() if not line.startswith("#") and
                            any(key in line for key in ("num_requests_running{", "num_requests_waiting{",
                                "kv_cache_usage_perc{", "num_preemptions_total{"))]
                    except Exception as exc:
                        sample["metrics_error"] = repr(exc)
                    for host in args.memory_hosts:
                        try:
                            sample["hosts"][host] = json.load(urllib.request.urlopen(
                                f"http://{host}:8082/memory", timeout=3))["host"]
                        except Exception as exc:
                            sample["hosts"][host] = {"error": repr(exc)}
                    log.write(json.dumps(sample) + "\n")
                    log.flush()
                    stop.wait(5)

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        try:
            wave = bench.wave(label, jobs)
        finally:
            stop.set()
            thread.join(timeout=20)
        rows = [json.loads((bench.OUT / (name + ".json")).read_text()) for name, _ in jobs]
        overlap = min(r["finished_at_unix"] for r in rows) - max(r["first_token_at_unix"] for r in rows)
        checks = []
        for row in rows:
            item = expected[row["name"]]
            got = first_codes(row["content"])
            checks.append({"name": row["name"], "expected": item["codes"], "got": got,
                "pass": got == item["codes"] and not row["error"],
                "input_tokens": row["usage"]["prompt_tokens"],
                "input_size_matches": row["usage"]["prompt_tokens"] == item["tokens"],
                "output_tokens": row["usage"]["completion_tokens"],
                "fixed_length_matches": row["usage"]["completion_tokens"] == args.output_tokens,
                "ttft_s": row["ttft_s"], "elapsed_s": row["elapsed_s"],
                "output_tok_s_approx_decode": row["output_tok_s_approx_decode"]})
        fresh = wave["metrics_delta"].get("request_prefill_kv_computed_tokens_sum") == wave["prompt_tokens"]
        preemptions = wave["metrics_delta"].get("num_preemptions_total", 0)
        grade = {"label": label, "concurrency": concurrency, "checks": checks,
            "full_context_decode_overlap_s": overlap, "all_input_tokens_computed": fresh,
            "preemptions": preemptions, "wall_s": wave["wall_s"],
            "aggregate_prefill_tok_s_wall": wave["aggregate_prompt_tok_s_wall"],
            "pass": all(c["pass"] and c["input_size_matches"] and c["fixed_length_matches"] for c in checks)
                    and overlap > 0 and fresh and preemptions == 0}
        grades.append(grade)
        (bench.OUT / "long-context-grades.json").write_text(json.dumps(grades, indent=2))
        print("GRADE " + json.dumps(grade), flush=True)
        if not grade["pass"]:
            raise SystemExit("Stopping before larger concurrency: previous capacity/quality check failed")


if __name__ == "__main__":
    main()
