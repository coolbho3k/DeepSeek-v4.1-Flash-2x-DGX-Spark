# SPDX-License-Identifier: AGPL-3.0-only
"""CPU-only prefix-survival EMA; does not change sampling or verification.

Inspired by the attributed GLM recipe's warmup/prefix/recovery design. Unlike
feeding a fabricated K on full short-prefix acceptance, retain censored tail
estimates and explicitly explore full K. One difficult request does not take
the minimum of the entire batch: optimize expected *batch* useful tokens.
"""
from dataclasses import dataclass, field
import math


@dataclass
class State:
    survival: list[float]
    steps: int = 0
    short_steps: int = 0


@dataclass
class PrefixEMA:
    lengths: tuple[int, ...] = (1, 2, 3, 4, 5)
    alpha: float = .25
    warmup: int = 4
    explore_after: int = 8
    margin: float = 1.
    state: dict[str, State] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if (not isinstance(self.lengths, tuple) or not self.lengths
                or self.lengths != tuple(sorted(set(self.lengths)))
                or any(type(k) is not int or not 1 <= k <= 5 for k in self.lengths)
                or isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or not 0 < self.alpha <= 1
                or type(self.warmup) is not int or self.warmup < 1
                or type(self.explore_after) is not int or self.explore_after < 1
                or isinstance(self.margin, bool) or not math.isfinite(self.margin) or self.margin < 0):
            raise ValueError('Invalid bounded EMA configuration')

    @property
    def maximum(self): return self.lengths[-1]

    def observe(self, request_id, drafted, accepted):
        if (not isinstance(request_id, str) or not request_id
                or type(drafted) is not int or not 0 <= drafted <= self.maximum
                or type(accepted) is not int or not 0 <= accepted <= drafted):
            raise ValueError('Invalid acceptance observation')
        if not drafted: return
        state = self.state.setdefault(request_id, State([1.]*self.maximum))
        for i in range(self.maximum):
            if i >= drafted and accepted == drafted:
                continue  # Full short prefix: no evidence about the tail.
            observed = float(i < accepted)
            state.survival[i] += self.alpha*(observed-state.survival[i])
        # Censoring with different observation ages can otherwise invert the
        # tail. A later prefix cannot survive more often than an earlier one.
        for i in range(1,self.maximum):
            state.survival[i] = min(state.survival[i],state.survival[i-1])
        state.steps += 1
        state.short_steps = 0 if drafted == self.maximum else state.short_steps+1

    def choose(self, request_ids, *, structured=(), costs=None):
        """Uniform verified prefix; costs optionally prices this exact batch.

        Cost values include the constant full-K draft cost plus verification
        and graph-padding cost, not only the marginal target kernel cost.
        Calls do not mutate state: scheduler retries cannot advance exploration.
        """
        request_ids=tuple(request_ids)
        if len(request_ids)>6 or len(set(request_ids))!=len(request_ids):
            raise ValueError('Use one explicit C1..C6 request batch')
        if not request_ids: return self.maximum
        if set(request_ids)&set(structured): return self.maximum
        states=[self.state.get(rid) for rid in request_ids]
        if any(s is None or s.steps<self.warmup or s.short_steps>=self.explore_after for s in states):
            return self.maximum
        if costs is not None:
            if (set(costs)!=set(self.lengths)
                    or any(isinstance(v,bool) or not math.isfinite(v) or v<=0 for v in costs.values())):
                raise ValueError('Need positive measured total-step costs for every prefix')
            scores={k:sum(1+sum(s.survival[:k]) for s in states)/costs[k] for k in self.lengths}
            # Prefer shorter work on an exact tie, not fabricated acceptance.
            return max(self.lengths,key=lambda k:(scores[k],-k))
        target=math.ceil(sum(sum(s.survival) for s in states)/len(states)+self.margin)
        return max((k for k in self.lengths if k<=target),default=self.lengths[0])

    def prune(self, live_ids):
        live=set(live_ids)
        self.state={key:value for key,value in self.state.items() if key in live}
