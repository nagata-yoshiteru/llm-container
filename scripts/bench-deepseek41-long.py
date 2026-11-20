#!/usr/bin/env python3
"""Retry one saved chat benchmark with a larger per-request native API budget.

Uses server-side chat tokenization and actual runtime context length. Does not
change server configuration. Generated code is saved for review, never executed.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def call(url, route, body=None, timeout=20):
    req = Request(url + route, None if body is None else json.dumps(body).encode(),
                  {'Content-Type': 'application/json'})
    return urlopen(req, timeout=timeout)


def reasoning_count(event, think_end_ids, thinking):
    """Count reasoning from generated IDs; the native metadata may report zero."""
    if not thinking:
        return 0
    ids = event.get('output_ids', [])
    if not ids or not think_end_ids:
        return None
    # Repetition aborts can trim text/completion_tokens but retain extra IDs.
    count = event.get('meta_info', {}).get('completion_tokens', len(ids))
    width = len(think_end_ids)
    for i in range(len(ids) - width + 1):
        if ids[i:i + width] == think_end_ids:
            return i if i < count and '</think>' in event.get('text', '') else None
    return count if '</think>' not in event.get('text', '') else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--max-new-tokens', type=int, default=32768)
    p.add_argument('--url', default='http://127.0.0.1:8910')
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--temperature', type=float, default=0)
    p.add_argument('--seed', type=int)
    args = p.parse_args()
    source = json.loads(args.source.read_text())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    marker = args.output_dir / 'run-start.json'
    if marker.exists(): p.error('Output directory already has a run')
    args.url = args.url.rstrip('/')
    with call(args.url, '/get_server_info') as r: info = json.load(r)
    model = info['served_model_name']
    with call(args.url, '/v1/tokenize', {
        'model': model, 'messages': [{'role': 'user', 'content': source['prompt']}],
        'chat_template_kwargs': {'thinking': source['thinking']},
    }) as r: tokenized = json.load(r)
    with call(args.url, '/v1/tokenize', {
        'model': model, 'prompt': '</think>', 'add_special_tokens': False,
    }) as r: think_end_ids = json.load(r)['tokens']
    # /v1/tokenize.max_model_len is the checkpoint limit, not the running limit.
    context = info['context_length']
    input_ids = tokenized['tokens']
    budget = min(args.max_new_tokens, context - len(input_ids) - 1)
    if budget <= 0: p.error('Prompt does not fit the running context')
    payload = dict(input_ids=input_ids, stream=True,
                   sampling_params=dict(temperature=args.temperature, max_new_tokens=budget,
                                        skip_special_tokens=False))
    if args.seed is not None:
        payload['sampling_params']['sampling_seed'] = args.seed
    save(marker, dict(timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
         model=model, source=str(args.source.resolve()), route='/generate',
         context_length=context, prompt_tokens=len(input_ids),
         requested_max_new_tokens=args.max_new_tokens, effective_max_new_tokens=budget,
         temperature=args.temperature, seed=args.seed,
         note='Same prompt and server chat tokenization; native generation avoids the chat-specific output cap.'))
    save(args.output_dir / 'request.json', payload)
    print(f'START: input={len(input_ids)}, output_budget={budget}, context={context}', flush=True)
    start = time.perf_counter()
    first = last = None
    previous_text = ''
    event = {}
    done = False
    error = None
    report_at = start
    try:
        with call(args.url, '/generate', payload, args.timeout) as response:
            for line in response:
                now = time.perf_counter()
                if now-start > args.timeout: raise TimeoutError('Total generation deadline exceeded')
                if not line.startswith(b'data:'): continue
                raw = line[5:].strip()
                if raw == b'[DONE]': done = True; break
                event = json.loads(raw)
                if 'error' in event: raise RuntimeError(event['error'])
                text = event.get('text', '')
                if text != previous_text:
                    if first is None: first = now
                    last = now
                    previous_text = text
                if now-report_at >= 30:
                    meta = event.get('meta_info', {})
                    print(f'RUNNING: {now-start:.1f}s, tokens={meta.get("completion_tokens")}, '
                          f'final_answer_started={"</think>" in text}', flush=True)
                    save(args.output_dir / 'progress.json', dict(elapsed_s=now-start, **event))
                    report_at = now
        if not done: raise RuntimeError('Incomplete native stream')
    except Exception as exc:
        error = repr(exc)
    elapsed = time.perf_counter()-start
    text = event.get('text', '')
    meta = event.get('meta_info', {})
    finish = meta.get('finish_reason') or {}
    if finish.get('type') == 'abort' and error is None:
        error = finish.get('message') or 'Server aborted generation'
    if '</think>' in text:
        reasoning, content = text.split('</think>', 1)
        content = content.removesuffix('<｜end▁of▁sentence｜>').strip()
    else:
        reasoning, content = text, ''
    count = meta.get('completion_tokens', 0)
    row = dict(name=source['name'], prompt=source['prompt'], thinking=source['thinking'],
               max_tokens=budget, route='/generate', error=error, finish_reason=finish.get('type'),
               usage=dict(prompt_tokens=meta.get('prompt_tokens'), completion_tokens=count,
                          reasoning_tokens=reasoning_count(event, think_end_ids, source['thinking'])),
               reasoning_tokens_method='IDs before </think> excluding delimiter; retained completion count if no closing delimiter',
               elapsed_s=elapsed, ttft_s=first-start if first is not None else None,
               decode_tok_s_estimate=(count-1)/(last-first)
               if last is not None and last>first and count>1 else None,
               reasoning=reasoning, content=content, native_meta_info=meta)
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', content, re.S)
    row['grade'] = {'status': 'pending_review' if blocks else 'missing_code'}
    if blocks: (args.output_dir / (source['name']+'.py')).write_text(blocks[-1]+'\n')
    save(args.output_dir / (source['name']+'.json'), row)
    save(args.output_dir / 'native-final.json', event)
    print(f'DONE: {count} tokens, {elapsed:.1f}s, finish={finish}, '
          f'code_found={bool(blocks)}, error={error}', flush=True)
    if error: raise SystemExit(1)


if __name__ == '__main__':
    main()
