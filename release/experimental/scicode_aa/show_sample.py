# SPDX-License-Identifier: AGPL-3.0-only
"""Show per-step model output stats and a reasoning excerpt for one scored SciCode problem."""
import glob
import sys

from inspect_ai.log import read_eval_log

E = '/home/emi/code/ds41/artifacts/evals/scicode'
problem, step, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]) if len(sys.argv) > 3 else 600
log = read_eval_log(glob.glob(f'{E}/runs/aa-logs/pass1/*.eval')[0], resolve_attachments=True)
sample = next(x for x in log.samples if str(x.id) == problem)
events = [e for e in sample.events if e.event == 'model']


def split(message):
    reasoning = text = ''
    for part in message.content if isinstance(message.content, list) else [message.content]:
        if hasattr(part, 'reasoning'):
            reasoning += part.reasoning or ''
        elif hasattr(part, 'text'):
            text += part.text or ''
        elif isinstance(part, str):
            text += part
    return reasoning, text


outputs = []
for i, event in enumerate(events, 1):
    choice = event.output.choices[0]
    reasoning, text = split(choice.message)
    usage = event.output.usage
    outputs.append((reasoning, text))
    print(f'call {i}: output_tokens={getattr(usage, "output_tokens", None)} '
          f'stop={choice.stop_reason} reasoning_chars={len(reasoning)} answer_chars={len(text)}'
          f'{" error=" + str(event.error)[:200] if getattr(event, "error", None) else ""}'
          f'{" retries=" + str(event.retries) if getattr(event, "retries", None) else ""}')
reasoning, text = outputs[step - 1]
print(f'\n--- {problem}.{step} reasoning, start ---\n{reasoning[:n]}\n...\n'
      f'--- {problem}.{step} reasoning, end ---\n{reasoning[-n:]}')
