# SPDX-License-Identifier: AGPL-3.0-only
"""Generate a chat-style token-frequency corpus disjoint from benchmark prompts.

Prompts cover prose, code in several languages, markdown structure, math,
data formats, and Chinese/Spanish/French/German/Japanese. None of the serving
benchmark prompts are used, so held-out benchmark replies can evaluate coverage.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request

PROMPTS = [
    'Write a short story about a lighthouse keeper who finds a message in a bottle.',
    'Summarize the causes and consequences of the French Revolution in a structured outline.',
    'Explain how vaccines train the immune system, with headings and bullet points.',
    'Compare TCP and UDP in a markdown table, then discuss when to use each.',
    'Write a Rust function that parses a CSV line with quoted fields, with tests.',
    'Implement binary search in C++ and explain edge cases.',
    'Write a JavaScript debounce function and show how to use it in a React component.',
    'Write a SQL query to find the top 5 customers by revenue per region, and explain it.',
    'Write a bash script that backs up a directory with rotation of the last 7 copies.',
    'Explain gradient descent with a worked numerical example and LaTeX formulas.',
    'Prove that the square root of 2 is irrational, step by step.',
    'Solve: a train leaves at 3pm at 60 km/h, another at 4pm at 80 km/h; when does the second catch up?',
    'Produce a JSON object describing a fictional library with books, authors and loans.',
    'Write a YAML configuration for a CI pipeline that tests a Python package on three versions.',
    'Give me a weekly meal plan for a vegetarian athlete as a markdown table.',
    'Draft a polite email declining a job offer while keeping the door open.',
    'Explain the difference between a process and a thread to a junior developer.',
    'Describe how a refrigerator works, using simple analogies.',
    'Write a haiku sequence about autumn in a city.',
    'List ten tips for improving sleep quality, each with a one-sentence rationale.',
    'Explain Kubernetes pods, deployments and services with an example manifest.',
    'Write a Go HTTP server with two routes and graceful shutdown.',
    'Explain the Monty Hall problem and why switching is better.',
    'Write a product description for an ergonomic office chair.',
    '用中文介绍中国古代四大发明，并说明它们的影响。',
    '请写一段Python代码实现快速排序，并用中文解释其时间复杂度。',
    '请用中文写一封感谢老师的信。',
    '解释一下什么是区块链，并举一个生活中的例子。',
    'Explica en español cómo funciona la fotosíntesis y por qué es importante.',
    'Escribe una receta de paella paso a paso.',
    'Explique en français les principales causes du changement climatique.',
    'Erkläre auf Deutsch, wie ein Elektromotor funktioniert.',
    '日本語で、桜の季節についての短いエッセイを書いてください。',
    'Write a regex that validates email addresses and explain each part.',
    'Explain the CAP theorem with concrete database examples.',
    'Write unit tests in pytest for a function that merges overlapping intervals.',
    'Describe the plot of a mystery novel set on a space station, in three acts.',
    'Explain big-O notation with examples of O(1), O(log n), O(n) and O(n^2).',
    'Write a TypeScript interface for a user profile and a function that validates it.',
    'Give step-by-step instructions to change a flat tire.',
    'Explain how HTTPS protects data, including certificates and key exchange.',
    'Write a limerick about a cat who learns to code.',
    'Explain the water cycle in a numbered list for middle school students.',
    'Write a Dockerfile for a Node.js app with a multi-stage build and explain it.',
    'Translate this into formal English and improve it: we gonna ship the thing next week maybe.',
    'Explain what a hash map is and implement one in Java with separate chaining.',
    'Discuss the pros and cons of remote work in a balanced essay.',
    'Explain the rules of chess castling and en passant with examples in algebraic notation.',
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', default='http://127.0.0.1:8889')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-tokens', type=int, default=320)
    a = p.parse_args()
    assert not a.output.exists()

    def ask(prompt):
        body = dict(model='deepseek-v41-flash-exl3', messages=[dict(role='user', content=prompt)],
                    max_tokens=a.max_tokens, temperature=1.0, top_p=0.95, seed=7,
                    chat_template_kwargs={'thinking': False})
        req = urllib.request.Request(a.base + '/v1/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=900) as r:
            d = json.load(r)
        return dict(prompt=prompt, reply=d['choices'][0]['message']['content'],
                    completion_tokens=d['usage']['completion_tokens'])

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(ask, PROMPTS))
    a.output.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(dict(prompts=len(rows), tokens=sum(r['completion_tokens'] for r in rows))))


if __name__ == '__main__':
    main()
