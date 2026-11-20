"""One synthetic near-limit retrieval check; run from the repository root."""
import argparse
import importlib.util
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

sys.path.insert(0, 'scripts')
spec = importlib.util.spec_from_file_location('bench', 'scripts/bench-deepseek41.py')
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--target-tokens', type=int, default=261120)
p.add_argument('--output-dir', type=Path, required=True)
opts = p.parse_args()
out = opts.output_dir
out.mkdir(exist_ok=True)
if (out / 'context-smoke.json').exists():
    p.error('Result already exists')
args = SimpleNamespace(url='http://127.0.0.1:8910', timeout=1200, output_dir=out)
with bench.request(args.url, '/get_server_info') as r:
    info = json.load(r)
args.model = info['served_model_name']
target = min(opts.target_tokens, info['context_length'] - 1024)
assert target > 1024
rng = random.Random(20261120)
vocab = 'forest river mountain stone blue red green winter summer spring north south east west house road water earth light night tree garden bird cloud rain lake sea wind moon sun book paper table chair gold silver black white field hill'.split()
words = rng.choices(vocab, k=target * 2)
n = target
expected = '72619384'
for _ in range(10):
    prompt = ('Retrieval check 20261120-L8R4. Read the background and locate the record.\n'
              + ' '.join(words[:n//2])
              + '\nRecord L8R4 has verification code ' + expected + '.\n'
              + ' '.join(words[n//2:n])
              + '\nWhat is the verification code for record L8R4? Reply with only its eight digits.')
    with bench.request(args.url, '/v1/tokenize', {
        'model': args.model, 'messages': [{'role': 'user', 'content': prompt}],
        'chat_template_kwargs': {'thinking': False},
    }, timeout=120) as r:
        count = json.load(r)['count']
    if target - 64 <= count <= target:
        break
    n += target - count - 16
else:
    raise RuntimeError('Unable to fit prompt')
bench.save(out / 'request-info.json', dict(prompt_tokens=count, context_length=info['context_length'],
    expected=expected, needle_depth_approx=0.5, max_tokens=32,
    note='One synthetic random-word retrieval case, not a full long-context quality benchmark.'))
print(f'START context smoke: {count} input tokens, 32 output budget', flush=True)
row = bench.stream(args, 'context-smoke', prompt, 32, thinking=False)
row['grade'] = dict(expected=expected, got=row['content'].strip(),
    passed=row['error'] is None and row['finish_reason']=='stop' and row['content'].strip()==expected)
cached = ((row.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens', 0) or 0
row['client_effective_prefill_tok_s'] = (count-cached)/row['ttft_s'] if row['ttft_s'] else None
bench.save(out / 'context-smoke.json', row)
print('RESULT', row['grade'], flush=True)
