# SPDX-License-Identifier: AGPL-3.0-only
"""Progress and scores for the Artificial-Analysis-style SciCode run (run on the worker)."""
import glob
import json
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

E = Path('/home/emi/code/ds41/artifacts/evals/scicode')
STATE = E / 'runs/aa-state'
AA = 149.33 / 288          # Artificial Analysis, 2026-09-14: 0.5185 (Epoch AI mirror)
SCORED = 288               # AA's denominator; the official harness divides by 291
TOTAL_GENERATED = 288      # steps the model writes (3 of 291 are supplied)
CAP = 3072 * 2**20
# Steps that fail unconditionally in the unfixed official harness (SciCode issue #59).
ISSUE59 = {f'13.{i}' for i in range(7, 16)} | {f'62.{i}' for i in range(2, 7)}


def read(path, default=''):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def pass_stats(n):
    root = E / f'runs/aa-pass{n}/openai-api-ds41-deepseek-v41-flash-exl3'
    generated = list((root / 'generated_code/with_background').glob('*.py'))
    results = {}
    for f in (root / 'evaluation_logs/with_background').glob('*.log'):
        results[f.stem] = read(f).splitlines()[0] if read(f) else '?'
    counts = {k: sum(v == k for v in results.values()) for k in ('pass', 'fail', 'time out', 'oom')}
    problems = {s.split('.')[0] for s in results}
    errors = 0
    for log in glob.glob(str(E / f'runs/aa-logs/pass{n}/*.eval')):
        try:
            from inspect_ai.log import read_eval_log
            errors += sum(1 for s in (read_eval_log(log).samples or []) if s.error)
        except Exception:
            pass
    issue59 = sum(1 for k, v in results.items() if k in ISSUE59 and v == 'pass')
    return dict(generated=len(generated), scored=len(results), problems=len(problems),
                errors=errors, issue59=issue59, **counts)


def started(n):
    for line in read(STATE / 'history').splitlines():
        if line.startswith(f'pass {n} started'):
            return datetime.fromisoformat(line.split()[-1].replace('Z', '+00:00'))
    return None


def server():
    try:
        m = urllib.request.urlopen('http://10.100.32.1:8888/metrics', timeout=5).read().decode()
        get = lambda k: sum(float(l.split()[-1]) for l in m.splitlines() if l.startswith(k))
        return get('vllm:num_requests_running'), get('vllm:generation_tokens_total')
    except Exception:
        return None, None


def main():
    now = datetime.now(timezone.utc)
    print(f'SciCode, Artificial Analysis method, reasoning_effort=max, max_tokens=393216, C4    {now:%Y-%m-%d %H:%M} UTC')
    print(f'status: {read(STATE / "status", "not started")}')
    finished = []
    for n in (1, 2, 3):
        s = pass_stats(n)
        if not s['generated'] and not s['scored']:
            print(f'pass {n}: not started'); continue
        done = (STATE / f'pass{n}.done').exists()
        rate = s['pass'] / s['scored'] if s['scored'] else 0
        line = (f'pass {n}: {"done" if done else "running"} | problems scored {s["problems"]}/65 | '
                f'steps written {s["generated"]}/{TOTAL_GENERATED} | steps scored {s["scored"]}: '
                f'pass {s["pass"]} fail {s["fail"]} timeout {s["time out"]} oom {s["oom"]} | '
                f'pass rate so far {100 * rate:.1f}%')
        if s['errors']:
            line += f' | sample errors {s["errors"]}'
        t0 = started(n)
        if t0 and not done and s['generated']:
            elapsed = (now - t0).total_seconds()
            eta = elapsed / s['generated'] * (TOTAL_GENERATED - s['generated'])
            line += f' | elapsed {elapsed / 3600:.1f} h, ~{eta / 3600:.1f} h left'
        print(line)
        if done:
            finished.append((s['pass'] / SCORED, (s['pass'] - s['issue59']) / SCORED))
    if finished:
        mean = sum(f for f, _ in finished) / len(finished)
        unfixed = sum(u for _, u in finished) / len(finished)
        print(f'score (subproblem pass@1, /288, mean of {len(finished)} pass(es)): {100 * mean:.2f}%   '
              f'AA: {100 * AA:.2f}%   difference {100 * (mean - AA):+.2f} pts')
        print(f'  if the unfixed official harness (issue #59, 14 steps forced to fail): {100 * unfixed:.2f}% '
              f'(difference {100 * (unfixed - AA):+.2f} pts)')
    else:
        print(f'score: pending first complete pass   AA: {100 * AA:.2f}%')
    try:
        st = subprocess.run(['docker', 'stats', '--no-stream', '--format', '{{.MemUsage}}', 'scicode-aa'],
                            capture_output=True, text=True, timeout=20).stdout.strip() or 'not running'
    except Exception:
        st = '?'
    avail = int(next(l.split()[1] for l in open('/proc/meminfo') if l.startswith('MemAvailable'))) // 1024
    running, gen = server()
    print(f'container memory {st} (cap 3 GiB) | worker MemAvailable {avail} MiB '
          f'(watchdog stops at 1024, serving at 512) | server requests running {running}')


if __name__ == '__main__':
    main()
