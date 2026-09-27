"""Where in the BF16 ranking do format-induced swaps happen? (stitched 16K contexts)"""
import sys, json, torch
sys.path.insert(0, '/work/release/experimental/indexer_fp4')  # run like analyze.py; writes /out/depth.json
import analyze as A
from collections import defaultdict
from pathlib import Path
from safetensors.torch import load_file
torch.set_num_threads(6)
proj = Path('/out/proj'); table = A.freqs_cis(16384)
records = sorted(p.stem for p in (proj / '20').glob('text-*.safetensors'))
corpora = defaultdict(list)
for r in records: corpora[r.split('-')[1]].append(r)
for c in corpora: corpora[c].sort(key=lambda r: int(r.split('-')[2]))
out = {}
for layer in (2, 20, 24, 36):
    ratio = A.RATIOS[layer]; res = defaultdict(list)
    for c, rs in corpora.items():
        k = torch.cat([load_file(str(proj / f'{A.KEY_SOURCE[layer]:02d}' / f'{r}.safetensors'))['k'] for r in rs])
        d = load_file(str(proj / f'{layer:02d}' / f'{rs[-1]}.safetensors'))
        local = torch.arange(0, 2048, 16); pos = local + 2048 * (len(rs) - 1)
        q = A.rope(d['q'][local], pos, table); kk = A.rope(k, torch.arange(k.size(0)) * ratio, table)
        s_ref = A.scores(q, d['w'][local], kk, pos, ratio)
        order = s_ref.argsort(-1, descending=True)
        rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(order.size(1)).expand_as(order))
        for name in ('mx', 'nv46'):
            fq, fk = A.FORMATS[name]
            s = A.scores(fq(q), d['w'][local], fk(kk), pos, ratio)
            chosen = s.topk(512, -1).indices
            r_chosen = rank.gather(1, chosen)             # BF16 rank of each kept position
            dropped_best = []                             # best BF16 rank that was dropped
            for i in range(len(pos)):
                kept = torch.zeros(k.size(0), dtype=torch.bool); kept[chosen[i]] = True
                miss = (~kept[order[i, :512]]).nonzero()
                if len(miss): dropped_best.append(int(miss[0]))
            res[name + '_best_dropped_rank'] += dropped_best
            res[name + '_worst_added_rank'] += r_chosen.max(1).values.tolist()
    q_ = lambda v, p: sorted(v)[int(p * (len(v) - 1))]
    out[layer] = {k: dict(p1=q_(v, .01), p10=q_(v, .1), median=q_(v, .5), p90=q_(v, .9)) for k, v in res.items()}
    print(layer, json.dumps(out[layer]), flush=True)
json.dump(out, open('/out/depth.json', 'w'), indent=1)
