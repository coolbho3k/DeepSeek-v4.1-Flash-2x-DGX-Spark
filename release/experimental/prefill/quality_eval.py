# SPDX-License-Identifier: AGPL-3.0-only
"""Collect teacher-forced prompt log-probabilities and greedy replies.

Run once against the control server and once against a candidate, then
compare with compare_quality.py. Documents are fixed concatenations of local
calibration-corpus token records (~6K tokens each); every token passes through
every attention layer. Each prompt is sent once per server.
"""
import argparse
import glob
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[3]
PROMPTS = [
    'Explain how a hash table handles collisions, with a short Python example.',
    'Write a haiku about winter, then explain its imagery.',
    'Summarize the causes of World War I in five bullet points.',
    'What is the derivative of x^3 * sin(x)? Show the steps.',
    'Translate into French: The library opens at nine and closes at six.',
    '用中文解释什么是光合作用。',
    'Explica en español qué es la inflación.',
    'Write a SQL query that returns the second highest salary per department.',
    'Give three tips for writing clear technical documentation.',
    'Describe the plot of Romeo and Juliet in one paragraph.',
    'Write a bash one-liner that counts lines in all .py files recursively.',
    'Why is the sky red at sunset? Answer briefly.',
]


def call(base, path, payload):
    req = urllib.request.Request(base + path, data=json.dumps(payload).encode(),
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)


def documents():
    records = []
    for path in sorted(glob.glob(str(ROOT / 'calibration/corpus-v1/records.jsonl'))):
        for line in open(path):
            r = json.loads(line)
            if r.get('tokens') and r.get('split') == 'calibration':
                records.append(r)
    by_source = {}
    for r in records:
        by_source.setdefault(r['source'], []).append(r)
    docs = []
    for source in ('wiki.utf8', 'code.utf8', 'c4.utf8', 'multilingual.utf8', 'technical.utf8'):
        rows = by_source.get(source, [])
        for i in range(0, min(len(rows), 6), 3):
            chunk = rows[i:i + 3]
            if len(chunk) == 3:
                docs.append(dict(source=source, ids=[r['id'] for r in chunk],
                                 tokens=[t for r in chunk for t in r['tokens']]))
    return docs[:6]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--concurrent', action='store_true', help='Send all documents at once (batch-shape reference)')
    a = p.parse_args()
    assert not a.output.exists()
    out = dict(base=a.base, documents=[], replies=[])
    docs = documents()
    def score(doc):
        return call(a.base, '/v1/completions', dict(model='deepseek-v41-flash-exl3', prompt=doc['tokens'],
                    max_tokens=1, temperature=0, prompt_logprobs=1))
    if a.concurrent:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(docs)) as pool:
            results = list(pool.map(score, docs))
    else:
        results = None
    for i, doc in enumerate(docs):
        d = results[i] if results else score(doc)
        rows = d['choices'][0]['prompt_logprobs'][1:]
        actual = []
        for tok, row in zip(doc['tokens'][1:], rows):
            entry = row[str(tok)]
            top = min(row.values(), key=lambda v: v['rank'])
            top_id = next(k for k, v in row.items() if v is top)
            actual.append([entry['logprob'], entry['rank'], int(top_id)])
        out['documents'].append(dict(source=doc['source'], records=doc['ids'], tokens=len(doc['tokens']),
                                     per_token=actual))
        print(json.dumps(dict(doc=doc['source'], tokens=len(doc['tokens']),
                              mean_nll=-sum(x[0] for x in actual) / len(actual))), flush=True)
    for prompt in ([] if a.concurrent else PROMPTS):
        d = call(a.base, '/v1/chat/completions', dict(model='deepseek-v41-flash-exl3',
                 messages=[dict(role='user', content=prompt)], max_tokens=256, temperature=0,
                 chat_template_kwargs={'thinking': False}))
        out['replies'].append(dict(prompt=prompt, reply=d['choices'][0]['message']['content']))
    a.output.write_text(json.dumps(out, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
