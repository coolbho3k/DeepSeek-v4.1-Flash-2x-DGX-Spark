# SPDX-License-Identifier: AGPL-3.0-only
"""Build the broad (off-policy) corpus: existing prompt/response pairs, tokenized
with the serving chat template via /tokenize.

The capture pass still records our target's own next-token distributions at
every position; only the conditioning text comes from other authors/models.
Output records match generate.py (prompt_token_ids, token_ids) with
on_policy=false. Needs pyarrow (drafter venv).
"""
import argparse
import collections
import gzip
import json
from pathlib import Path
import random
import urllib.request

import pyarrow.parquet as pq

RAW = Path(__file__).resolve().parents[3] / 'artifacts/drafter-data/raw'
MODEL = 'deepseek-v41-flash-exl3'


def tokenize(base, messages, generation_prompt):
    body = dict(model=MODEL, messages=messages, add_generation_prompt=generation_prompt,
                chat_template_kwargs={'thinking': False})
    req = urllib.request.Request(base + '/tokenize', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)['tokens']


def pairs(rng):
    out = []
    t = pq.read_table(RAW / 'mlabonne__open-perfectblend/data/train-00003-of-00006.parquet').to_pylist()
    for r in rng.sample(t, 12000):
        c = r['conversations']
        if len(c) >= 2 and c[0]['from'] == 'human' and c[1]['from'] == 'gpt':
            out.append(('perfectblend', 'mixed', None, c[0]['value'], c[1]['value']))
    for r in rng.sample(pq.read_table(RAW / 'CohereLabs__aya_dataset/data/train-00000-of-00001.parquet',
                                      columns=['inputs', 'targets', 'language']).to_pylist(), 9000):
        out.append(('aya', 'multilingual', r['language'], r['inputs'], r['targets']))
    trees = []
    with gzip.open(RAW / 'OpenAssistant__oasst2/2023-11-05_oasst2_ready.trees.jsonl.gz', 'rt') as f:
        for line in f:
            p = json.loads(line)['prompt']
            replies = [x for x in p.get('replies') or [] if x.get('role') == 'assistant']
            if replies:
                best = min(replies, key=lambda x: x.get('rank') if x.get('rank') is not None else 99)
                trees.append(('oasst2', 'multilingual', p.get('lang'), p['text'], best['text']))
    out += rng.sample(trees, min(5000, len(trees)))
    mc = [json.loads(l) for l in open(RAW / 'ise-uiuc__Magicoder-OSS-Instruct-75K/data-oss_instruct-decontaminated.jsonl')]
    out += [('magicoder', 'code', r.get('lang'), r['problem'], r['solution']) for r in rng.sample(mc, 5000)]
    cf = [json.loads(l) for l in open(RAW / 'm-a-p__CodeFeedback-Filtered-Instruction/CodeFeedback-Filtered-Instruction.jsonl')]
    out += [('codefeedback', 'code', r.get('lang'), r['query'], r['answer']) for r in rng.sample(cf, 5000)]
    om = pq.read_table(RAW / 'nvidia__OpenMathReasoning/data/additional_problems-00000-of-00001.parquet',
                       columns=['problem', 'generated_solution']).to_pylist()
    om = [r for r in om if r['generated_solution']]
    out += [('openmath', 'math_reasoning', None, r['problem'], r['generated_solution']) for r in rng.sample(om, min(4000, len(om)))]
    wc = pq.read_table(RAW / 'allenai__WildChat-1M/data/train-00007-of-00014.parquet',
                       columns=['conversation', 'language', 'toxic', 'redacted']).to_pylist()
    wc = [r for r in wc if not r['toxic'] and not r['redacted'] and len(r['conversation']) >= 2
          and r['conversation'][1]['role'] == 'assistant']
    out += [('wildchat', 'chat', r['language'], r['conversation'][0]['content'], r['conversation'][1]['content'])
            for r in rng.sample(wc, 12000)]
    return [x for x in out if x[3] and x[4] and len(x[3]) <= 8000 and 16 <= len(x[4]) <= 30000]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', default='http://127.0.0.1:8888')
    p.add_argument('--seed', type=int, default=43)
    a = p.parse_args()
    rng = random.Random(a.seed)
    items = pairs(rng)
    rng.shuffle(items)
    counts, total = collections.Counter(), 0
    with a.output.open('x') as out:
        for i, (source, category, lang, prompt, reply) in enumerate(items):
            user = [dict(role='user', content=prompt)]
            try:
                head = tokenize(a.base, user, True)
                full = tokenize(a.base, user + [dict(role='assistant', content=reply)], False)
            except Exception as e:
                print(json.dumps(dict(skip=i, error=str(e)[:200])), flush=True)
                continue
            if full[:len(head)] != head or len(full) <= len(head):
                counts['template_mismatch'] += 1
                continue
            rid = f'broad-{source}-{i:06d}'
            out.write(json.dumps(dict(id=rid, source=source, category=category, language=lang, thinking=False,
                                      temperature=None, on_policy=False, finish_reason='stop',
                                      prompt_token_ids=head, token_ids=full[len(head):]), ensure_ascii=False) + '\n')
            counts[source] += 1
            total += len(full)
            if (i + 1) % 2000 == 0:
                print(json.dumps(dict(done=i + 1, tokens=total, counts=counts)), flush=True)
    print(json.dumps(dict(final=True, records=sum(v for k, v in counts.items() if k != 'template_mismatch'),
                          tokens=total, counts=counts)), flush=True)


if __name__ == '__main__':
    main()
