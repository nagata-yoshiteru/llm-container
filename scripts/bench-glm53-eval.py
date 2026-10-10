#!/usr/bin/env python3
"""GLM cold-prefill, concurrent decode, and baseline quality suite.

Generates and saves code, but does NOT execute generated code. Review it first.
Uses only Python's standard library. Run on an otherwise idle endpoint.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
import uuid

from glm53_eval_tasks import MATH, MATH_SUFFIX, CODE, CODE_SUFFIX

MODEL = "glm-5.3-flash-uncensored"
API = "http://127.0.0.1:8910"
OUT = None


def metrics():
    text = urlopen(API + "/metrics", timeout=15).read().decode()
    totals = {}
    for line in text.splitlines():
        if line.startswith("#") or not line:
            continue
        label, value = line.rsplit(" ", 1)
        name = label.split("{", 1)[0].removeprefix("vllm:")
        if name.endswith(("_sum", "_count", "_total")) or name in (
            "num_requests_running", "num_requests_waiting",
        ):
            totals[name] = totals.get(name, 0) + float(value)
    return totals


def post(route, body):
    req = Request(API + route, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.load(urlopen(req, timeout=600))


def stream(name, body, route="/v1/chat/completions"):
    body = {"model": MODEL, "stream": True, "stream_options": {"include_usage": True}, **body}
    req = Request(API + route, json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = time.perf_counter()
    progress_at = start
    first = last = None
    reasoning, content = [], []
    usage, finish = None, None
    done = False
    error = None
    try:
        with urlopen(req, timeout=7200) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                raw = line[5:].strip()
                if raw == b"[DONE]":
                    done = True
                    break
                event = json.loads(raw)
                if "error" in event:
                    raise RuntimeError(event["error"])
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    rc = delta.get("reasoning") or delta.get("reasoning_content") or ""
                    ct = delta.get("content") or choice.get("text") or ""
                    reasoning.append(rc)
                    content.append(ct)
                    if rc or ct:
                        last = time.perf_counter()
                        first = first if first is not None else last
                    if choice.get("finish_reason"):
                        finish = choice["finish_reason"]
                now = time.perf_counter()
                if now - progress_at >= 60:
                    print(f"RUNNING {name}: {now - start:.0f}s (stream receiving)", flush=True)
                    progress_at = now
        if not done or usage is None:
            raise RuntimeError("Incomplete stream or missing usage")
    except Exception as exc:
        error = repr(exc)
    elapsed = time.perf_counter() - start
    tokens = (usage or {}).get("completion_tokens", 0)
    row = {"name": name, "usage": usage, "elapsed_s": elapsed,
           "ttft_s": first - start if first is not None else None,
           "output_tok_s_e2e": tokens / elapsed,
           "output_tok_s_approx_decode": (tokens - 1) / (last - first)
           if last is not None and last > first and tokens > 1 else None,
           "finish_reason": finish, "error": error,
           "reasoning": "".join(reasoning), "content": "".join(content)}
    if name in MATH:
        answers = re.findall(r"\\boxed\{([^}]*)\}", row["content"])
        got = answers[-1].replace(",", "").strip() if answers else None
        row["grade"] = {"expected": MATH[name][1], "got": got,
                        "pass": error is None and got == MATH[name][1]}
    elif name in CODE:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", row["content"], re.S)
        row["grade"] = {"status": "pending code review and execution"}
        if blocks:
            (OUT / (name + "-generated.py")).write_text(blocks[-1] + "\n")
    (OUT / (name + ".json")).write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n")
    print(f"DONE {name}: {tokens} tokens, {elapsed:.1f}s, {tokens / elapsed:.2f} tok/s, "
          f"finish={finish}, error={error}", flush=True)
    return row


def wave(label, jobs):
    before = metrics()
    if before.get("num_requests_running", 0) or before.get("num_requests_waiting", 0):
        raise RuntimeError("Server has other requests; refusing overlapping benchmark")
    print("START " + label, flush=True)
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        rows = list(pool.map(lambda job: stream(*job), jobs))
    wall = time.perf_counter() - start
    for _ in range(30):
        after = metrics()
        count = after.get("request_prefill_time_seconds_count", 0) - before.get("request_prefill_time_seconds_count", 0)
        if count >= len(jobs):
            break
        time.sleep(0.2)
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in after}
    output = sum((row["usage"] or {}).get("completion_tokens", 0) for row in rows)
    prompt = sum((row["usage"] or {}).get("prompt_tokens", 0) for row in rows)
    prefill_s = delta.get("request_prefill_time_seconds_sum", 0)
    decode_s = delta.get("request_decode_time_seconds_sum", 0)
    computed = delta.get("request_prefill_kv_computed_tokens_sum", 0)
    result = {"label": label, "concurrency": len(jobs), "wall_s": wall,
              "prompt_tokens": prompt, "completion_tokens": output,
              "aggregate_output_tok_s_e2e": output / wall,
              "aggregate_prompt_tok_s_wall": prompt / wall,
              "prefill_new_kv_tok_s_per_request_weighted": computed / prefill_s if prefill_s else None,
              "decode_tok_s_per_request_weighted": (output - len(rows)) / decode_s if decode_s else None,
              "metrics_delta": delta, "metrics_request_count_matches": count == len(jobs),
              "requests": [{k: v for k, v in row.items() if k not in ("content", "reasoning")} for row in rows]}
    (OUT / (label + "-wave.json")).write_text(json.dumps(result, indent=2) + "\n")
    if any(row["error"] for row in rows):
        raise RuntimeError(f"{label}: request errors; results saved")
    if count != len(jobs):
        raise RuntimeError(f"{label}: metrics count {count} != {len(jobs)}; results may be contaminated")
    return result


def main():
    global API, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=API)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--phase", choices=["profile", "quality", "retrieval"], required=True)
    args = parser.parse_args()
    API, OUT = args.url.rstrip("/"), args.output_dir
    OUT.mkdir(parents=True, exist_ok=True)
    marker = OUT / (args.phase + "-start.json")
    if marker.exists():
        parser.error("This phase already has results; use a new output directory")
    marker.write_text(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(),
                                 "phase": args.phase, "api": API, "model": MODEL,
                                 "note": "Decode streaming rate is approximate with speculative chunks. "
                                 "Prefill server interval includes scheduling stalls and first token, not pure GPU time."}, indent=2))
    if args.phase == "profile":
        for rep in range(3):
            jobs = []
            for i in range(6):
                prompt = ("Explain how a relational database executes a SQL query, from parsing "
                          "through optimization to execution. Give concrete examples.") if i % 2 == 0 else (
                          "Write a Python LRU cache with O(1) get and put operations. Explain "
                          "the invariants and cover capacity zero and replacement of existing keys.")
                jobs.append((f"decode-c6-r{rep}-{i}", {
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256, "min_tokens": 256, "ignore_eos": True,
                    "temperature": 0, "seed": 42, "chat_template_kwargs": {"reasoning_effort": "max"},
                }))
            wave(f"decode-c6-r{rep}", jobs)
        corpus = "\n".join(
            f"Record {i:06d}: warehouse {(i * 17) % 997}, inventory {(i * 7919) % 100003}, "
            f"quantity {(i * 13) % 211}. Confirm delivery before updating the inventory ledger."
            for i in range(8000))
        tokenized = post("/tokenize", {"model": MODEL, "prompt": corpus})
        tokens = tokenized["tokens"]
        if len(tokens) < 131072:
            raise RuntimeError("Corpus too short")
        (OUT / "prefill-corpus.json").write_text(json.dumps({
            "generator": "8000 deterministic inventory records; see bench-glm53-eval.py",
            "sha256": hashlib.sha256(corpus.encode()).hexdigest(), "tokens": len(tokens),
            "cache_policy": "unique cache_salt for every request; verify computed KV token metric",
        }, indent=2))
        for length, concurrency, repeats in [(1024, 1, 3), (8192, 1, 3), (32768, 1, 3),
                                               (131072, 1, 1), (8192, 2, 3), (8192, 6, 3)]:
            for rep in range(repeats):
                label = f"prefill-n{length}-c{concurrency}-r{rep}"
                jobs = [(label + f"-{i}", {"prompt": tokens[:length],
                         "max_tokens": 1, "min_tokens": 1, "ignore_eos": True,
                         "temperature": 0, "seed": 42, "cache_salt": uuid.uuid4().hex},
                         "/v1/completions") for i in range(concurrency)]
                wave(label, jobs)
    elif args.phase == "quality":
        def job(name):
            prompt = MATH[name][0] + MATH_SUFFIX if name in MATH else CODE[name][0] + CODE_SUFFIX
            return name, {"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": 32768, "temperature": 1.0, "top_p": 0.95,
                          "chat_template_kwargs": {"reasoning_effort": "max"}}
        wave("quality-single-M4", [job("M4_powmod")])
        wave("quality-concurrent-six", [job(name) for name in [
            "M1_aime06", "M2_aime99", "M3_digitsum", "C1_inversions", "C2_domino", "C3_minwindow"]])
    else:
        expected = {"RED": "Cobalt-73419-Orchid", "GREEN": "Quartz-58263-Finch", "BLUE": "Maple-91647-Lantern"}

        def make_prompt(count):
            lines = [f"Record {i:06d}: warehouse {(i * 17) % 997}, inventory {(i * 7919) % 100003}, "
                     f"quantity {(i * 13) % 211}. Confirm delivery before updating the inventory ledger."
                     for i in range(count)]
            for fraction, (label, value) in zip((0.1, 0.5, 0.9), expected.items()):
                lines[int(count * fraction)] += f" AUDIT NOTE: the exact verification code for {label} is {value}."
            return ("Read these reference records and remember the three audit verification codes.\n\n" +
                    "\n".join(lines) + "\n\nReturn a JSON object mapping RED, GREEN, and BLUE to their exact "
                    "verification codes from the records. Return only the JSON object.")

        count = 4000
        for _ in range(4):
            prompt = make_prompt(count)
            tokenized = post("/tokenize", {"model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "chat_template_kwargs": {"reasoning_effort": "max"}})
            actual = tokenized["count"]
            if 131072 <= actual <= 132096:
                break
            count = max(1, int(count * 131584 / actual))
        if not 131072 <= actual <= 132096:
            raise RuntimeError(f"Could not size retrieval prompt: {actual}")
        (OUT / "retrieval-input.json").write_text(json.dumps({"prompt": prompt, "tokens": actual,
                                                            "expected": expected}, indent=2))
        result = wave("retrieval-128k", [("retrieval-128k", {
            "messages": [{"role": "user", "content": prompt}], "max_tokens": 2048,
            "temperature": 0, "seed": 42, "cache_salt": uuid.uuid4().hex,
            "chat_template_kwargs": {"reasoning_effort": "max"}})])
        row = json.loads((OUT / "retrieval-128k.json").read_text())
        text = row["content"].strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
        try:
            got = json.loads(text)
        except ValueError:
            got = None
        grade = {"expected": expected, "got": got, "pass": got == expected,
                 "input_tokens": row["usage"]["prompt_tokens"],
                 "ttft_s": row["ttft_s"], "elapsed_s": row["elapsed_s"]}
        (OUT / "retrieval-grade.json").write_text(json.dumps(grade, indent=2) + "\n")
        print("RETRIEVAL " + json.dumps(grade), flush=True)


if __name__ == "__main__":
    main()
