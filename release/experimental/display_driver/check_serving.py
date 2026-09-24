# SPDX-License-Identifier: AGPL-3.0-only
"""Small mixed-driver HTTP canary, not a quality or throughput qualification."""
import argparse
import json
from pathlib import Path
import time
import urllib.request


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not a.output.parent.is_dir():
        raise ValueError('Use a fresh report path in an existing directory')
    with urllib.request.urlopen('http://127.0.0.1:8888/health', timeout=5) as r:
        assert r.status == 200
    rows = []
    for name, prompt, limit in (
        ('arithmetic', 'What is 7 times 8? Answer with only the number.', 16),
        ('prose', 'Write a vivid short scene in plain English about a librarian opening a village library on a rainy morning. Use continuous prose, no lists or headings.', 256),
    ):
        payload = dict(model='deepseek-v41-flash-exl3',
            messages=[dict(role='user', content=prompt)], temperature=0, seed=41,
            reasoning_effort='none', max_tokens=limit, stream=True,
            stream_options=dict(include_usage=True))
        req = urllib.request.Request('http://127.0.0.1:8888/v1/chat/completions',
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'})
        start = time.monotonic()
        first = last = None
        content = reasoning = ''
        usage = None
        finish = None
        with urllib.request.urlopen(req, timeout=120) as response:
            for raw in response:
                line = raw.decode().strip()
                if not line.startswith('data: '):
                    continue
                if line == 'data: [DONE]':
                    break
                event = json.loads(line[6:])
                if event.get('error'):
                    raise RuntimeError(event['error'])
                usage = event.get('usage') or usage
                for choice in event.get('choices', []):
                    delta = choice.get('delta', {})
                    text = delta.get('content') or ''
                    thought = delta.get('reasoning') or delta.get('reasoning_content') or ''
                    if text or thought:
                        last = time.monotonic()
                        first = first or last
                    content += text
                    reasoning += thought
                    finish = choice.get('finish_reason') or finish
        row = dict(name=name, elapsed_seconds=time.monotonic()-start,
            time_to_first_text_seconds=first-start if first else None,
            streamed_text_seconds=last-first if first else None,
            content=content, reasoning=reasoning, usage=usage, finish_reason=finish)
        if not content or not usage or (name == 'arithmetic' and content.strip() != '56'):
            raise RuntimeError('Canary returned unexpected content or no token usage')
        rows.append(row)
        print(json.dumps(row), flush=True)
    with a.output.open('x') as stream:
        json.dump(dict(status='short_http_canaries_pass', results=rows,
            quality_qualified=False, long_context_tested=False,
            performance_benchmark=False), stream, indent=2)


if __name__ == '__main__':
    main()
