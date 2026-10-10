#!/usr/bin/env python3
"""Summarize bench-glm53.py and bench-glm53-eval.py artifacts without rerunning."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    root = args.directory
    fixed = json.loads((root / "fixed-output-c1-c2.json").read_text())
    waves = [json.loads(p.read_text()) for p in sorted(root.glob("*-wave.json"))]
    summary = {"fixed_output": fixed["summary"], "output_by_concurrency": [],
               "cold_prefill": [], "quality": [],
               "notes": ["Output rates count reasoning and final output tokens.",
                         "E2E rates include queue and prefill; streaming decode rates are approximate.",
                         "Prefill phase includes scheduling stalls and first token, not pure GPU kernel time.",
                         "128K prefill is a throughput/stability test, not a long-context retrieval-quality test.",
                         "Both dense precision and MTP k changed from the baseline."]}
    if args.baseline:
        baseline = json.loads(args.baseline.read_text())
        for row in summary["fixed_output"]:
            old = next(s for s in baseline["summary"] if
                       (s["concurrency"], s["kind"]) == (row["concurrency"], row["kind"]))
            row["baseline_tok_s"] = old["median_per_request_tok_s"]
            row["improvement_percent"] = (row["median_per_request_tok_s"] / row["baseline_tok_s"] - 1) * 100
    for concurrency in (1, 2, 6):
        if concurrency < 6:
            samples = [s for s in fixed["samples"] if s["concurrency"] == concurrency]
            groups = [s["rows"] for s in samples]
            walls = [s["wall_s"] for s in samples]
            rate_key = "e2e_tok_s"
        else:
            samples = [s for s in waves if s["label"].startswith("decode-c6-")]
            groups = [s["requests"] for s in samples]
            walls = [s["wall_s"] for s in samples]
            rate_key = "output_tok_s_e2e"
        aggregate = [sum(r["usage"]["completion_tokens"] for r in group) / wall
                     for group, wall in zip(groups, walls)]
        rows = [r for group in groups for r in group]
        if not aggregate:
            continue
        summary["output_by_concurrency"].append({
            "concurrency": concurrency, "waves": len(samples),
            "median_per_request_e2e_tok_s": statistics.median(r[rate_key] for r in rows),
            "median_aggregate_e2e_tok_s": statistics.median(aggregate),
            "aggregate_min_max": [min(aggregate), max(aggregate)],
            "median_ttft_s": statistics.median(r["ttft_s"] for r in rows),
            "max_ttft_s": max(r["ttft_s"] for r in rows),
        })
    prefill = [s for s in waves if s["label"].startswith("prefill-")]
    keys = sorted({(s["prompt_tokens"] // s["concurrency"], s["concurrency"]) for s in prefill})
    for length, concurrency in keys:
        samples = [s for s in prefill if s["prompt_tokens"] == length * concurrency and s["concurrency"] == concurrency]
        all_fresh = all(s["metrics_delta"]["request_prefill_kv_computed_tokens_sum"] == s["prompt_tokens"] for s in samples)
        summary["cold_prefill"].append({
            "input_tokens_per_request": length, "concurrency": concurrency,
            "repeats": len(samples), "all_tokens_newly_computed": all_fresh,
            "median_server_prefill_tok_s_per_request": statistics.median(s["prefill_new_kv_tok_s_per_request_weighted"] for s in samples),
            "median_aggregate_wall_tok_s": statistics.median(s["aggregate_prompt_tok_s_wall"] for s in samples),
            "median_ttft_s": statistics.median(r["ttft_s"] for s in samples for r in s["requests"] if r["ttft_s"] is not None),
            "median_wave_wall_s": statistics.median(s["wall_s"] for s in samples),
        })
    grades_path = root / "code-grades.json"
    grades = json.loads(grades_path.read_text()) if grades_path.exists() else {}
    for name in ["M4_powmod", "M1_aime06", "M2_aime99", "M3_digitsum", "C1_inversions", "C2_domino", "C3_minwindow"]:
        path = root / (name + ".json")
        if not path.exists():
            continue
        row = json.loads(path.read_text())
        summary["quality"].append({k: v for k, v in row.items() if k not in ("reasoning", "content")})
        if name in grades:
            summary["quality"][-1]["grade"] = grades[name]
    for s in waves:
        if s["label"].startswith("quality-"):
            summary[s["label"]] = {k: v for k, v in s.items() if k != "requests"}
    retrieval = root / "retrieval-grade.json"
    if retrieval.exists():
        summary["retrieval_128k"] = json.loads(retrieval.read_text())
    summary["all_wave_metric_counts_match"] = all(s["metrics_request_count_matches"] for s in waves)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
