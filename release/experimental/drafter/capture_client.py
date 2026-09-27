# SPDX-License-Identifier: AGPL-3.0-only
"""Drive the capture kit: prefill each generated sequence once, collect its features.

For every record with prompt_token_ids + token_ids, send a 1-token completion
whose prompt is the exact token sequence (unique cache_salt, so prefix caching
cannot skip positions). The capture kit writes <capture>/<request id>/*.pt on
the head; this client moves that folder to <data>/<record id>/ and adds
meta.json. Resumable; standard library only.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import urllib.request
import uuid

MODEL = 'deepseek-v41-flash-exl3'


def split_of(record_id, heldout_mod):
    if record_id.startswith('gate1-'):
        return 'gate1'
    return 'heldout' if int(hashlib.sha256(record_id.encode()).hexdigest(), 16) % heldout_mod == 0 else 'train'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records', type=Path, nargs='+', required=True)
    p.add_argument('--capture-dir', type=Path, required=True, help='host path of the kit /cache/drafter-capture')
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--base', default='http://127.0.0.1:8889')
    p.add_argument('--concurrency', type=int, default=3)
    p.add_argument('--heldout-mod', type=int, default=50)
    p.add_argument('--max-tokens-total', type=int, default=24576)
    p.add_argument('--limit', type=int, default=0)
    a = p.parse_args()
    a.data.mkdir(parents=True, exist_ok=True)
    done = {q.name for q in a.data.iterdir() if (q / 'meta.json').exists()}
    todo = []
    for path in a.records:
        for line in path.open():
            r = json.loads(line)
            n = len(r['prompt_token_ids']) + len(r['token_ids'])
            if r['id'] not in done and 8 <= n <= a.max_tokens_total and r.get('finish_reason') in ('stop', 'length', 'tool_calls', None):
                todo.append(r)
    if a.limit:
        todo = todo[:a.limit]
    lock = threading.Lock()
    stats = dict(ok=0, err=0, tokens=0, start=time.time())

    def work(r):
        request_id = 'cap-' + r['id']
        tokens = r['prompt_token_ids'] + r['token_ids']
        body = dict(model=MODEL, prompt=tokens, max_tokens=1, temperature=0.0, cache_salt=uuid.uuid4().hex)
        req = urllib.request.Request(a.base + '/v1/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json', 'X-Request-Id': request_id})
        try:
            with urllib.request.urlopen(req, timeout=3600) as resp:
                json.load(resp)
            found = [q for q in a.capture_dir.iterdir() if request_id in q.name]
            if len(found) != 1:
                raise RuntimeError(f'expected one capture folder for {request_id}, found {len(found)}')
            folder = found[0]
            time.sleep(0)  # prefill rows are flushed when the request leaves the batch; nudge if needed
            for _ in range(240):
                if sum(json.loads(q.read_text())['rows'] for q in folder.glob('*.json')) >= len(tokens):
                    break
                nudge = urllib.request.Request(a.base + '/v1/completions', data=json.dumps(dict(
                    model=MODEL, prompt=[0, 1], max_tokens=1, temperature=0.0)).encode(),
                    headers={'Content-Type': 'application/json', 'X-Request-Id': 'nudge'})
                urllib.request.urlopen(nudge, timeout=120).read()
                time.sleep(0.5)
            positions = sorted(int(q.stem) for q in folder.glob('*.pt'))
            if not positions or positions[0] != 0:
                raise RuntimeError(f'{request_id}: capture does not start at position 0: {positions[:3]}')
            dest = a.data / r['id']
            shutil.move(str(folder), dest)
            meta = dict(id=r['id'], prompt_len=len(r['prompt_token_ids']), total_len=len(tokens),
                        split=split_of(r['id'], a.heldout_mod), source=r.get('source'), category=r.get('category'),
                        language=r.get('language'), thinking=r.get('thinking'), temperature=r.get('temperature'),
                        on_policy=r.get('on_policy', True))
            (dest / 'meta.json').write_text(json.dumps(meta))
        except Exception as e:
            with lock:
                stats['err'] += 1
                print(json.dumps(dict(id=r['id'], error=f'{type(e).__name__}: {e}'[:400])), flush=True)
            return
        with lock:
            stats['ok'] += 1
            stats['tokens'] += len(tokens)
            if stats['ok'] % 50 == 0:
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
