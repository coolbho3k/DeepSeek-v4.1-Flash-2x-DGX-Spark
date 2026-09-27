# SPDX-License-Identifier: AGPL-3.0-only
"""Generate on-policy responses from the serving target (standard library only).

With --capture-dir/--data (capture kit), each response's decode-time features
are moved into the dataset as it finishes: no separate prefill capture pass.

Reads the prompt pool in order, skips ids already present in the output, and
appends one JSON line per finished request with the exact prompt and response
token ids (return_token_ids), so the capture pass re-feeds precisely what the
target produced. Safe to stop and restart at any time.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import threading
import time
import urllib.error
import urllib.request

MODEL = 'deepseek-v41-flash-exl3'


def split_of(record_id, heldout_mod=50):
    if record_id.startswith('gate1-'):
        return 'gate1'
    return 'heldout' if int(hashlib.sha256(record_id.encode()).hexdigest(), 16) % heldout_mod == 0 else 'train'


def nudge(base):
    """One tiny request so the capture hook flushes buffers of requests that just finished."""
    body = dict(model=MODEL, prompt=[0, 1], max_tokens=1, temperature=0.0)
    req = urllib.request.Request(base + '/v1/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'X-Request-Id': 'nudge'})
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def captured_rows(folder):
    """Rows on disk, from the JSON sidecar the hook writes next to each immutable chunk."""
    return sum(json.loads(q.read_text())['rows'] for q in folder.glob('*.json') if q.name != 'meta.json')


def collect(base, capture_dir, data, row, deadline=120):
    """Move this request's decode-time capture into the dataset once all its rows are on disk."""
    rid = 'cap-' + row['id']
    need = len(row['prompt_token_ids']) + len(row['token_ids']) - 1   # the last token is never an input row
    start = time.time()
    while True:
        found = [q for q in capture_dir.iterdir() if rid in q.name]
        if len(found) == 1 and captured_rows(found[0]) >= need:
            break
        if time.time() - start > deadline:
            raise RuntimeError(f'capture for {rid} incomplete after {deadline}s')
        nudge(base)
        time.sleep(0.5)
    dest = data / row['id']
    shutil.move(str(found[0]), dest)
    meta = dict(id=row['id'], prompt_len=len(row['prompt_token_ids']), total_len=need + 1, split=split_of(row['id']),
                source=row['source'], category=row['category'], language=row.get('language'),
                thinking=row['thinking'], temperature=row['temperature'], on_policy=True, decode_capture=True)
    (dest / 'meta.json').write_text(json.dumps(meta))


def request(base, item, timeout, capture=False):
    body = dict(model=MODEL, messages=item['messages'], max_tokens=item['max_tokens'],
                temperature=item['temperature'], top_p=item['top_p'], return_token_ids=True,
                chat_template_kwargs={'thinking': item['thinking']})
    if item.get('tools'):
        body['tools'] = item['tools']
    if item.get('reasoning_effort'):
        body['reasoning_effort'] = item['reasoning_effort']
    headers = {'Content-Type': 'application/json'}
    if capture:
        headers['X-Request-Id'] = 'cap-' + item['id']
    req = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(body).encode(), headers=headers)
    started = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    c = d['choices'][0]
    return dict(id=item['id'], source=item['source'], category=item['category'], language=item.get('language'),
                thinking=item['thinking'], reasoning_effort=item.get('reasoning_effort'),
                temperature=item['temperature'], top_p=item['top_p'], tools=bool(item.get('tools')),
                prompt_token_ids=d['prompt_token_ids'], token_ids=c['token_ids'],
                finish_reason=c['finish_reason'], seconds=round(time.time() - started, 2),
                content=c['message'].get('content'), reasoning=c['message'].get('reasoning'),
                tool_calls=c['message'].get('tool_calls'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--base', default='http://127.0.0.1:8888')
    p.add_argument('--concurrency', type=int, default=6)
    p.add_argument('--limit', type=int, default=0, help='stop after this many new responses (0 = all)')
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--capture-dir', type=Path, help='host path of the capture kit /cache/drafter-capture')
    p.add_argument('--data', type=Path, help='dataset root for decode-time captures')
    a = p.parse_args()
    done = set()
    if a.output.exists():
        with a.output.open() as f:
            done = {json.loads(line)['id'] for line in f if line.strip()}
    todo = [json.loads(line) for line in a.prompts.open()]
    todo = [t for t in todo if t['id'] not in done]
    if a.limit:
        todo = todo[:a.limit]
    lock = threading.Lock()
    stats = dict(ok=0, err=0, tokens=0, start=time.time())
    errors = a.output.with_suffix('.errors.jsonl')

    def work(item):
        try:
            row = request(a.base, item, a.timeout, capture=a.capture_dir is not None)
            if a.capture_dir is not None:
                collect(a.base, a.capture_dir, a.data, row)
        except Exception as e:  # keep going; failures are retried on the next run
            with lock, errors.open('a') as f:
                f.write(json.dumps(dict(id=item['id'], error=f'{type(e).__name__}: {e}'[:500])) + '\n')
                stats['err'] += 1
            return
        with lock:
            with a.output.open('a') as f:
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
            stats['ok'] += 1
            stats['tokens'] += len(row['token_ids'])
            if stats['ok'] % 25 == 0:
                rate = stats['tokens'] / (time.time() - stats['start'])
                print(json.dumps(dict(done=stats['ok'], errors=stats['err'], tokens=stats['tokens'],
                                      tok_per_s=round(rate, 1), remaining=len(todo) - stats['ok'] - stats['err'])),
                      flush=True)

    print(json.dumps(dict(already_done=len(done), queued=len(todo))), flush=True)
    with concurrent.futures.ThreadPoolExecutor(a.concurrency) as pool:
        list(pool.map(work, todo))
    print(json.dumps(dict(final=stats)), flush=True)


if __name__ == '__main__':
    main()
