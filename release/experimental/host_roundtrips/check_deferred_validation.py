# SPDX-License-Identifier: AGPL-3.0-only
"""Single-GPU check of deferred graph-validation draining (no TP, no model).

Exercises GraphOwner replay bookkeeping with the in-graph summary simulated:
clean replays pass; a device error flag or a nonzero peer summary raises at
drain and poisons the owner; replaying again without a drain is refused.
The cross-rank all-reduce itself is exercised by the full two-Spark boot.
"""
import sys

import torch

sys.path.insert(0, '/opt/ds41-serving')
from ds41 import graph_validation as v

MESSAGES = ((1, 'bad slot'), (2, 'bad page'))


def owner_with_graph(error_value, peer_value):
    owner = v.GraphOwner(torch.device('cuda', 0))
    errors = torch.zeros(4, dtype=torch.int32, device='cuda')
    graph = torch.cuda.CUDAGraph()
    with owner.execution(capture_only=True):
        owner.capture_only = True
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
            errors.fill_(error_value)
            v.check_flags(errors, MESSAGES)
            owner.summary.zero_()
            owner.summary[1:2].fill_(peer_value)
        owner.rank, owner.deferred = 0, True
    return owner, graph


def replay(owner, graph):
    with owner.execution(capture_only=False):
        graph.replay()


def main():
    results = {}
    owner, graph = owner_with_graph(0, 0)
    for _ in range(3):
        replay(owner, graph)
        v.drain_pending()
    results['clean'] = 'pass'
    replay(owner, graph)
    try:
        owner._defer(torch.cuda.current_stream())
    except RuntimeError:
        results['undrained_replay_refused'] = 'pass'
    v.drain_pending()
    for label, (err, peer, expect) in dict(own_error=(2, 0, 'bad page'),
                                           peer_error=(0, 1, 'peer')).items():
        o, g = owner_with_graph(err, peer)
        replay(o, g)
        try:
            v.drain_pending()
        except ValueError as error:
            assert expect in str(error), error
            assert o.failed
            results[label] = 'raised_and_poisoned'
        else:
            raise AssertionError(label + ' not raised')
        try:
            replay(o, g)
        except RuntimeError:
            results[label + '_poisoned_refuses_replay'] = 'pass'
    print(results)
    assert len(results) == 6


if __name__ == '__main__':
    main()
