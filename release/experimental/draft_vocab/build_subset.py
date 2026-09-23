# SPDX-License-Identifier: AGPL-3.0-only
"""Choose a frequency-ranked draft vocabulary from the local calibration corpus.

Each TP rank owns half of the vocabulary-parallel output head. The subset
takes the same number of most frequent tokens from each half so that the
drafter's local logits have equal widths for the existing TP all-gather.
Only the draft proposal distribution changes; target verification and
rejection sampling remain exact with respect to the target distribution.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VOCAB = 129280
HALF = VOCAB // 2
# Special tokens the drafter must always be able to propose.
ALWAYS = (0, 1, 2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--per-rank', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert 1024 <= a.per_rank <= HALF and not a.output.exists()
    counts = collections.Counter()
    sources = {}
    for path in sorted((ROOT / 'calibration').glob('corpus-*/records.jsonl')):
        raw = path.read_bytes()
        sources[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
        for line in raw.splitlines():
            tokens = json.loads(line).get('tokens')
            if tokens:
                counts.update(tokens)
    total = sum(counts.values())
    ranks = []
    for rank in (0, 1):
        lo, hi = rank * HALF, (rank + 1) * HALF
        forced = [t for t in ALWAYS if lo <= t < hi]
        ranked = sorted((t for t in counts if lo <= t < hi and t not in forced),
                        key=lambda t: (-counts[t], t))
        # Pad with the lowest unseen ids so every rank has exactly per_rank rows.
        chosen = set(forced + ranked[:a.per_rank - len(forced)])
        filler = (t for t in range(lo, hi) if t not in chosen)
        while len(chosen) < a.per_rank:
            chosen.add(next(filler))
        ranks.append(sorted(chosen))
    covered = sum(counts[t] for r in ranks for t in r)
    result = dict(format='ds41_draft_vocab_v1', vocab=VOCAB, per_rank=a.per_rank, ranks=ranks,
                  corpus_tokens=total, in_sample_coverage=covered / total, corpus_sha256=sources)
    a.output.write_text(json.dumps(result) + '\n')
    print(json.dumps(dict(output=str(a.output), per_rank=a.per_rank, in_sample_coverage=covered / total,
                          sha256=hashlib.sha256(a.output.read_bytes()).hexdigest())))


if __name__ == '__main__':
    main()
