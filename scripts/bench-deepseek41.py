#!/usr/bin/env python3
"""SGLang streaming speed and pass@1 checks. Stdlib only; outputs raw responses.

Code is saved first. Review it before running --phase grade-code.
Prefill is a client-side estimate, not a GPU kernel measurement.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import os
from pathlib import Path
import random
import re
import resource
import subprocess
import sys
import tempfile
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import uuid
from zoneinfo import ZoneInfo

from glm53_eval_tasks import MATH, CODE, MATH_SUFFIX, CODE_SUFFIX

MATH = dict(MATH, M5_derangement_cycles=(
    "How many permutations of {1,2,...,10} have no fixed points and exactly three cycles "
    "in their disjoint cycle decomposition?", "303660"), M6_recurrence_gcd=(
    "Let a_0=0, a_1=1, and a_n=3*a_(n-1)-a_(n-2) for n>=2. "
    "Find gcd(a_2026, a_1518).", "3"))
CODE = dict(CODE, C4_dynamic_connectivity=(
    "Implement offline_connectivity(n: int, operations: list[tuple[str,int,int]]) -> list[bool]. "
    "The initially empty undirected multigraph has vertices 0..n-1. Operations are "
    "('add',u,v), ('remove',u,v), or ('query',u,v). Add increments the multiplicity of an edge; "
    "remove decrements it (a copy is guaranteed to exist). Query asks whether u and v are "
    "connected, including a vertex to itself. Return answers in query order. Self-loops and "
    "parallel edges are allowed. n and the number of operations are at most 200000. "
    "Use an efficient offline algorithm, e.g. a segment tree of time intervals and rollback DSU, "
    "with O((n+q) log(q) log(n)) time or better.", r'''
import collections, random
rng = random.Random(271828)
def reference(n, ops):
    edges = collections.Counter(); answers = []
    for kind,u,v in ops:
        edge = tuple(sorted((u,v)))
        if kind == 'add': edges[edge] += 1
        elif kind == 'remove': edges[edge] -= 1
        else:
            graph = [[] for _ in range(n)]
            for (a,b),copies in edges.items():
                if copies: graph[a].append(b); graph[b].append(a)
            seen = {u}; stack = [u]
            while stack:
                for w in graph[stack.pop()]:
                    if w not in seen: seen.add(w); stack.append(w)
            answers.append(v in seen)
    return answers
assert offline_connectivity(1, []) == []
for _ in range(80):
    n = rng.randrange(1,10); active = []; ops = []
    for step in range(100):
        u,v = rng.randrange(n),rng.randrange(n)
        kind = rng.randrange(3)
        if kind == 0: active.append((u,v)); ops.append(('add',u,v))
        elif kind == 1 and active:
            u,v = active.pop(rng.randrange(len(active))); ops.append(('remove',v,u))
        else: ops.append(('query',u,v))
    assert offline_connectivity(n,ops) == reference(n,ops), ops
n = 30000
ops = [('add',i,i+1) for i in range(n-1)] + [('query',0,n-1)]
for i in range(n-1):
    ops.extend([('remove',i,i+1),('query',0,n-1),('add',i,i+1),('query',0,n-1)])
assert offline_connectivity(n,ops) == [True] + [False,True]*(n-1)
'''))


def request(api, route, body=None, timeout=20):
    data = None if body is None else json.dumps(body).encode()
    req = Request(api + route, data, {"Content-Type": "application/json"})
    return urlopen(req, timeout=timeout)


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stream(args, name, prompt, max_tokens, thinking=False, **extra):
    body = dict(model=args.model, messages=[dict(role="user", content=prompt)],
                max_tokens=max_tokens, temperature=0, stream=True,
                stream_options={"include_usage": True},
                chat_template_kwargs={"thinking": thinking}, **extra)
    start = time.perf_counter()
    first = last = None
    reasoning, content, chunk_times = [], [], []
    usage, finish, error, done = None, None, None, False
    try:
        with request(args.url, "/v1/chat/completions", body, args.timeout) as response:
            for line in response:
                if time.perf_counter() - start > args.timeout:
                    raise TimeoutError("Request exceeded total time budget")
                if not line.startswith(b"data:"): continue
                raw = line[5:].strip()
                if raw == b"[DONE]": done = True; break
                event = json.loads(raw)
                if "error" in event: raise RuntimeError(str(event["error"]))
                if event.get("usage"): usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta", {})
                    rc = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    ct = delta.get("content") or ""
                    reasoning.append(rc); content.append(ct)
                    if rc or ct:
                        last = time.perf_counter()
                        if first is None: first = last
                        chunk_times.append(last-start)
                    if choice.get("finish_reason"): finish = choice["finish_reason"]
        if not done or usage is None: raise RuntimeError("Incomplete stream or missing token usage")
    except HTTPError as exc:
        error = f"HTTP {exc.code}: {exc.read().decode(errors='replace')}"
    except Exception as exc:
        error = repr(exc)
    elapsed = time.perf_counter()-start
    tokens = (usage or {}).get("completion_tokens", 0)
    row = dict(name=name, prompt=prompt, max_tokens=max_tokens, thinking=thinking,
               usage=usage, finish_reason=finish, error=error, elapsed_s=elapsed,
               ttft_s=first-start if first is not None else None,
               decode_tok_s_estimate=(tokens-1)/(last-first)
               if last is not None and last>first and tokens>1 else None,
               output_tok_s_e2e=tokens/elapsed, chunk_times_s=chunk_times,
               reasoning="".join(reasoning), content="".join(content))
    if name in MATH:
        answers = re.findall(r"\\boxed\{([^{}]+)\}", row["content"])
        got = answers[-1].replace(",", "").strip() if answers else None
        row["grade"] = dict(expected=MATH[name][1], got=got,
                            passed=error is None and finish == "stop" and got == MATH[name][1])
    elif name in CODE:
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", row["content"], re.S)
        row["grade"] = {"status": "pending_review" if blocks else "missing_code"}
        if blocks: (args.output_dir / (name+".py")).write_text(blocks[-1]+"\n")
    save(args.output_dir / (name+".json"), row)
    print(f"DONE {name}: {tokens} tokens / {elapsed:.1f}s; ttft={row['ttft_s']}; "
          f"decode={row['decode_tok_s_estimate']}; finish={finish}; error={error}; "
          f"grade={row.get('grade')}", flush=True)
    return row


def grade_code(args):
    allowed = {"collections", "heapq", "bisect", "functools", "math", "typing", "itertools"}
    forbidden = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars",
                 "getattr", "setattr", "delattr", "input", "breakpoint", "help", "exit", "quit"}
    functions = dict(C1_inversions="count_inversions", C2_domino="tilings",
                     C3_minwindow="min_window", C4_dynamic_connectivity="offline_connectivity")
    grades = {}
    for name, (_, tests) in CODE.items():
        if args.only and name not in args.only.split(','): continue
        path = args.output_dir / (name+".py")
        if not path.exists(): grades[name] = dict(passed=False, error="missing code"); continue
        code = path.read_text()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    if any(a.name not in allowed for a in node.names): raise ValueError("import outside allowlist")
                if isinstance(node, ast.ImportFrom):
                    if node.level or node.module not in allowed or any(a.name == '*' for a in node.names):
                        raise ValueError("import outside allowlist")
                if isinstance(node, ast.Name) and (node.id in forbidden or node.id.startswith('__')):
                    raise ValueError("unsafe name: "+node.id)
                if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
                    raise ValueError("dunder attribute")
            # AST checks are a guardrail, not a security sandbox: review code first.
            with tempfile.TemporaryDirectory(prefix="deepseek-grade-") as tmp:
                root = Path(tmp)
                (root / 'solution.py').write_text(code)
                fn = functions[name]
                extra = ""
                if name == "C1_inversions":
                    extra = "\nn=200000\nassert count_inversions(list(range(n,0,-1)))==n*(n-1)//2\nassert count_inversions([7]*n)==0\n"
                if name == "C3_minwindow": extra = "\nassert min_window('abc','')==''\nassert min_window('','a')==''\n"
                (root / 'judge.py').write_text(
                    "import runpy, random\nrandom.seed(12345)\n"
                    f"candidate=runpy.run_path('solution.py')\n{fn}=candidate[{fn!r}]\n"+tests+extra)
                def limits():
                    resource.setrlimit(resource.RLIMIT_CPU, (20,20))
                    resource.setrlimit(resource.RLIMIT_AS, (768*1024**2,768*1024**2))
                    resource.setrlimit(resource.RLIMIT_FSIZE, (1024**2,1024**2))
                    resource.setrlimit(resource.RLIMIT_CORE, (0,0))
                start = time.perf_counter()
                result = subprocess.run([sys.executable, '-I', '-S', 'judge.py'], cwd=root,
                    env={"PATH": os.defpath, "LANG": "C.UTF-8"}, capture_output=True, text=True,
                    timeout=25, preexec_fn=limits)
                grades[name] = dict(passed=result.returncode==0, returncode=result.returncode,
                    elapsed_s=time.perf_counter()-start, stdout=result.stdout[-4000:], stderr=result.stderr[-4000:])
        except Exception as exc:
            grades[name] = dict(passed=False, error=repr(exc))
        print(name, grades[name], flush=True)
    save(args.output_dir / 'code-grades.json', grades)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8910')
    parser.add_argument('--model', default='deepseek-v4.1-flash-1017-uncensored')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--phase', choices=['speed','quality','grade-code'], required=True)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--timeout', type=int, default=1200)
    parser.add_argument('--wait-ready', type=int, default=900)
    parser.add_argument('--only', help='comma-separated quality/grading task IDs')
    parser.add_argument('--concurrencies', default='1,2', help='comma-separated speed phase concurrency levels')
    args = parser.parse_args()
    try:
        concurrencies = [int(x) for x in args.concurrencies.split(',')]
        if not concurrencies or any(x < 1 for x in concurrencies): raise ValueError()
    except ValueError:
        parser.error('--concurrencies must contain positive integers')
    if args.only and set(args.only.split(',')) - (set(MATH) | set(CODE)):
        parser.error('Unknown quality task ID in --only')
    args.url = args.url.rstrip('/')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == 'grade-code': grade_code(args); return
    marker = args.output_dir / (args.phase+'-start.json')
    if marker.exists(): parser.error('This phase already ran; use another output directory')
    deadline = time.monotonic()+args.wait_ready
    while True:
        try:
            with request(args.url, '/health', timeout=5): pass
            break
        except Exception:
            if time.monotonic()>deadline: raise RuntimeError('API not ready')
            print('Waiting for /health...', flush=True); time.sleep(10)
    save(marker, dict(timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(), model=args.model,
                     phase=args.phase, max_tokens=args.max_tokens, temperature=0,
                     note='Client TTFT includes queuing and first decode. Speculative decoding emits chunks; '
                          'decode tok/s is approximate. Prefill estimates are not GPU-only throughput.'))
    for route in ['/get_server_info', '/v1/models']:
        try:
            with request(args.url, route) as response:
                save(args.output_dir/(route.strip('/').replace('/','-')+'.json'), json.load(response))
        except Exception as exc: print(route, repr(exc), flush=True)
    if args.phase == 'quality':
        chosen = set(args.only.split(',')) if args.only else set(MATH)|set(CODE)
        for name, (prompt, _) in {**MATH, **CODE}.items():
            if name not in chosen: continue
            suffix = MATH_SUFFIX if name in MATH else CODE_SUFFIX + (
                '\nUse only standard Python, with imports from collections, heapq, bisect, functools, '
                'math, typing, or itertools if needed. Do not use sys or external libraries.')
            row = stream(args, name, prompt+suffix, args.max_tokens, thinking=True)
            if row['error']: raise RuntimeError(row['error'])
    else:
        warmup = stream(args, 'warmup', 'What is 13*17? Reply with the integer only.', 32)
        if warmup['error'] or warmup['content'].strip()!='221': raise RuntimeError('Warmup failed')
        rows = []
        words = 'forest river mountain stone blue red green winter summer autumn spring north south east west'.split()
        for target in [1024,4096,8192]:
            for rep in range(2):
                rng = random.Random(target*10+rep)
                prompt = uuid.uuid4().hex+'\n'+ ' '.join(rng.choices(words,k=target))+'\nReply only OK.'
                row = stream(args, f'prefill-{target}-r{rep}', prompt, 1)
                if row['error']: raise RuntimeError(row['error'])
                usage = row['usage']; cached = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
                row['cached_prompt_tokens'] = cached
                row['prefill_tok_s_client_estimate'] = (usage['prompt_tokens']-(cached or 0))/row['ttft_s']
                save(args.output_dir/(row['name']+'.json'), row); rows.append(row)
        for concurrency in concurrencies:
            for rep in range(2):
                start = time.perf_counter()
                def run(i):
                    prompt=uuid.uuid4().hex+'\nExplain how a database recovers safely after a crash. Give a detailed technical explanation.'
                    return stream(args,f'decode-c{concurrency}-r{rep}-{i}',prompt,256,ignore_eos=True,min_tokens=256)
                with ThreadPoolExecutor(max_workers=concurrency) as pool: wave=list(pool.map(run,range(concurrency)))
                elapsed=time.perf_counter()-start
                if any(r['error'] for r in wave): raise RuntimeError('Decode request failed')
                result=dict(concurrency=concurrency,elapsed_s=elapsed,
                    aggregate_output_tok_s_e2e=sum(r['usage']['completion_tokens'] for r in wave)/elapsed,
                    request_names=[r['name'] for r in wave])
                save(args.output_dir/f'decode-c{concurrency}-r{rep}-wave.json',result)
        print('Speed phase complete.',flush=True)


if __name__ == '__main__':
    main()
