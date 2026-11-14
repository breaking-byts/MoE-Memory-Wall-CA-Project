"""Two-stream timeline simulator: GPU compute stream + one DRAM->HBM copy link.

What it models (per decode step, per MoE layer l, batch of B tokens in lock-step):

  GPU   [attention+gate of l] -> routing of l known -> [expert e1][expert e2]...
  link  demand fetches of l's missing experts (highest priority) | prefetches

HBM expert cache
  `capacity` experts, partitioned per layer (capacity // L slots per layer, the
  remainder to the first layers), as in Mixtral-offloading's per-layer LRU.  A
  layer that routes more experts than its slots (batch > 1) temporarily stages
  the extra ones and is trimmed back (LRU) after it has executed.  An expert must
  be resident before the GPU can run it; while the GPU waits for a copy it is
  idle = exposed expert-loading stall.

Policies
  on-demand   misses are fetched only after the router has chosen them.
              reuse=False reproduces the SSD-offloading framework (every routed
              expert is re-copied every step); reuse=True keeps the LRU cache.
  prefetch    when layer l has been routed (lookahead d >= 1; ST-MoE timing) or
              when layer l's attention starts (d = 0; APEX timing), the predictor
              gives probabilities for layer l+d and the budget rule picks a set.
              Non-resident experts of the set are queued on the link at low
              priority.  Demand fetches overtake queued prefetches; an in-flight
              copy is not pre-empted unless `preempt` is on.  Queued prefetches still waiting when their
              target layer is routed are promoted (if routed) or cancelled.
              Routing itself is never changed (correctness-preserving).
  Prefetcher options (ablation):
    guided_replacement  a prefetch evicts the resident expert of the target layer
                        with the lowest predicted probability, and only if that
                        probability is below both the incoming expert's and 0.5
                        (a likely-needed expert is never displaced) -- instead of
                        LRU, which lets prefetches evict experts that are about to
                        hit.  Without it, predicted experts are never evicted by a
                        prefetch.  For batch B the probability that any of the B
                        tokens needs expert e is 1 - prod_b (1 - p_b(e)).
    p_min               do not prefetch experts below this probability.
    slack_aware         experts beyond the top-K (the adaptive budget's extras) are
                        only prefetched while the link can still deliver them before
                        the target layer is routed (estimated from the link backlog
                        and the measured layer time), or if p >= p_conf.
    feedback            feedback-directed throttling (after Srinath et al., HPCA 2007):
                        an EMA of prefetch accuracy (a prefetch is right if its
                        expert is routed at the target layer; cancelled ones and
                        ones evicted or aborted before their target count as wrong) sets
                        the aggressiveness -- acc >= fb_hi: the budget's extra
                        experts are allowed; fb_lo <= acc < fb_hi: top-K only;
                        acc < fb_lo: only experts with p >= p_conf.
    preempt             a demand miss aborts an in-flight prefetch at the next
                        projection boundary if it is known to be wrong (its target is
                        the layer being routed and does not need it) or unlikely
                        (p < 0.5), so a wrong prefetch delays a demand by <= 1/3 of
                        an expert copy.

Per-op latencies: attention/gate time per step from the simulator CSV rows,
expert load/exec times from costs.py (validated against the same CSV).
All times in seconds.
"""
from dataclasses import dataclass, field
import heapq
import numpy as np

from . import costs
from .predictors import Context, select


@dataclass
class Policy:
    name: str = "on-demand"
    capacity: int = 64                 # experts resident in HBM (all layers)
    reuse: bool = True                 # keep experts across steps
    predictor: object = None
    lookahead: int = 1
    budget: tuple = ("fixed", 2)
    guided_replacement: bool = False
    p_min: float = 0.0
    slack_aware: bool = False
    p_conf: float = 0.8
    preempt: bool = False              # demand may abort an in-flight prefetch at a chunk boundary
    chunks: int = 3                    # an expert is copied as 3 projections (gate, up, down)
    feedback: bool = False             # accuracy-feedback throttling (see below)
    fb_hi: float = 0.7
    fb_lo: float = 0.4


@dataclass
class Entry:
    ready: float
    last_use: float
    prefetched: bool
    used: bool = False
    target: tuple = None               # (step, layer) a prefetch was issued for
    judged: bool = False               # prefetch already scored right/wrong


@dataclass(order=True)
class Req:
    prio: int
    seq: int
    key: tuple = field(compare=False)
    issue: float = field(compare=False)
    target: tuple = field(compare=False)
    demand: bool = field(compare=False)
    p: float = field(default=1.0, compare=False)
    alive: bool = field(default=True, compare=False)


class Sim:
    def __init__(self, R, H, tok0, nonexp_layer_s, plat, policy, spec=None, model=costs.MIXTRAL):
        self.R, self.plat, self.pol, self.model = R, plat, policy, model
        self.steps, self.L, self.B, self.K = R.shape
        self.E = model.experts
        self.nonexp = nonexp_layer_s                       # [steps] seconds per layer
        self.load = costs.expert_load_s(plat, model)
        self.ctx = Context(H, tok0, spec, self.E)
        base, extra = divmod(policy.capacity, self.L)
        self.slots = [base + (1 if l < extra else 0) for l in range(self.L)]
        assert min(self.slots) >= 1, "capacity must give every layer at least one slot"
        self.cache = [dict() for _ in range(self.L)]      # per layer: expert -> Entry
        self.pending, self.seq, self.link_free = [], 0, 0.0
        self.queued = {}                                   # key -> Req (not started)
        self.pinned = set()
        self.pred_p = {}                                   # (s, lt) -> probability vector
        self.protect = {}                                  # (s, lt) -> predicted keys
        self.layer_time = None                             # EMA of measured layer time
        self.inflight = None                               # (req, start, end) of the last copy
        self.now = (-1, -1)                                # (step, layer) being routed
        self.acc = 0.5                                     # feedback: EMA of prefetch accuracy
        self.bytes_moved = 0.0
        self.st = dict(demand_fetch=0, prefetch_fetch=0, hits=0, inflight_hits=0, misses=0,
                       prefetch_used=0, prefetch_on_target=0, prefetch_unused_evicted=0, prefetch_dropped=0,
                       prefetch_cancelled=0, prefetch_not_admitted=0, prefetch_aborted=0,
                       accesses=0)

    # ---------------------------------------------------------------- cache
    def resident(self, key):
        return key[1] in self.cache[key[0]]

    def entry(self, key):
        return self.cache[key[0]][key[1]]

    def _protected(self, l):
        out = set()
        for (s, lt), keys in self.protect.items():
            if lt == l:
                out |= {k[1] for k in keys}
        return out

    def _feedback(self, useful):
        self.acc = 0.95 * self.acc + 0.05 * (1.0 if useful else 0.0)

    def _evict(self, l, e):
        en = self.cache[l].pop(e)
        if en.prefetched and not en.used:
            self.st["prefetch_unused_evicted"] += 1
            if not en.judged:                            # evicted before its target ran
                self._feedback(False)

    def _reserve(self, t, req):
        """Find a slot in the target layer's partition for req (at time t)."""
        l = req.key[0]
        part = self.cache[l]
        if len(part) < self.slots[l]:
            return True
        cand = [e for e, en in part.items() if en.ready <= t and (l, e) not in self.pinned]
        if req.demand:
            if not cand:
                return True                                # demand may over-subscribe (staging)
            v = min(cand, key=lambda e: part[e].last_use)  # LRU
        elif self.pol.guided_replacement and req.target in self.pred_p:
            # evict the least likely resident of the target layer, but only if it is
            # less likely than the incoming expert (predicted ones included)
            if not cand:
                return False
            p = self.pred_p[req.target]
            v = min(cand, key=lambda e: (p[e], part[e].last_use))
            if p[v] >= min(req.p, 0.5):                    # never evict a likely-needed expert
                return False
        else:
            prot = self._protected(l)                      # plain LRU, never evict a predicted expert
            cand = [e for e in cand if e not in prot]
            if not cand:
                return False
            v = min(cand, key=lambda e: part[e].last_use)
        self._evict(l, v)
        return True

    def _trim(self, l):
        part = self.cache[l]
        prot = self._protected(l)
        while len(part) > self.slots[l]:
            cand = [e for e, en in part.items() if (l, e) not in self.pinned and e not in prot
                    and en.ready <= 1e30]
            if not cand:
                break
            self._evict(l, min(cand, key=lambda e: part[e].last_use))

    # ---------------------------------------------------------------- link
    def _push(self, key, t, target, demand, p=1.0):
        self.seq += 1
        r = Req(0 if demand else 1, self.seq, key, t, target, demand, p)
        heapq.heappush(self.pending, r)
        self.queued[key] = r
        return r

    def _commit_next(self, horizon):
        """Start the highest-priority eligible request if it starts before horizon."""
        while self.pending and not self.pending[0].alive:
            heapq.heappop(self.pending)
        live = [r for r in self.pending if r.alive]
        if not live:
            return False
        start = max(self.link_free, min(r.issue for r in live))
        if start >= horizon:
            return False
        r = min(x for x in live if x.issue <= start)
        r.alive = False
        self.queued.pop(r.key, None)
        if not self._reserve(start, r):
            self.st["prefetch_dropped"] += 1
            return True
        end = start + self.load
        self.link_free = end
        self.cache[r.key[0]][r.key[1]] = Entry(ready=end, last_use=start, prefetched=not r.demand,
                                               target=None if r.demand else r.target)
        self.st["demand_fetch" if r.demand else "prefetch_fetch"] += 1
        self.bytes_moved += self.model.expert_bytes
        self.inflight = (r, start, end)
        return True

    def _preempt(self, t, need):
        """Abort the in-flight prefetch at the next chunk boundary after t."""
        if not (self.pol.preempt and self.inflight):
            return
        r, start, end = self.inflight
        if r.demand or end <= t or r.key in need:
            return
        if r.target != self.now and r.p >= 0.5:          # a likely-correct prefetch for a later layer
            return
        chunk = self.load / self.pol.chunks
        boundary = start + np.ceil((t - start) / chunk - 1e-12) * chunk
        if boundary >= end - 1e-15:
            return
        self.cache[r.key[0]].pop(r.key[1], None)
        self.link_free = boundary
        self.bytes_moved -= self.model.expert_bytes * (end - boundary) / self.load
        self.st["prefetch_fetch"] -= 1
        self.st["prefetch_aborted"] += 1
        if r.target == self.now:
            self._feedback(False)
        self.inflight = None

    def _commit_until(self, horizon):
        while self._commit_next(horizon):
            pass

    def _backlog(self, t):
        """Link time already promised at time t (in-flight + queued requests)."""
        live = sum(1 for r in self.pending if r.alive)
        return max(self.link_free - t, 0.0) + live * self.load

    # ---------------------------------------------------------------- prefetch
    def _issue(self, s, lt, d, t):
        pol = self.pol
        if pol.predictor is None or s >= self.steps:
            return
        miss = np.ones(self.E)
        chosen = set()
        for b in range(self.B):
            p = pol.predictor.probs(self.ctx, s, lt, b, d)
            if p is None:
                continue
            miss *= 1.0 - np.clip(p, 0, 1)
            chosen.update(int(e) for e in select(p, self.K, pol.budget))
        P = 1.0 - miss                                     # P(expert needed by any slot of the batch)
        self.pred_p[(s, lt)] = P
        order = sorted(chosen, key=lambda e: -P[e])
        self.protect[(s, lt)] = {(lt, e) for e in order}
        n_slots = None
        if pol.slack_aware:
            lead = max(d, 1) * (self.layer_time or 0.0)
            n_slots = int(max(0.0, lead - self._backlog(t)) // self.load) + 1
        n = 0
        for rank, e in enumerate(order):
            k = (lt, e)
            if self.resident(k) or k in self.queued:
                continue
            if P[e] < pol.p_min:
                self.st["prefetch_not_admitted"] += 1
                continue
            extra = rank >= self.K * self.B                    # beyond the top-K per token
            if (n_slots is not None and extra and n >= n_slots and P[e] < pol.p_conf):
                self.st["prefetch_not_admitted"] += 1   # extra expert, no link slack
                continue
            if pol.feedback and P[e] < pol.p_conf and (
                    self.acc < pol.fb_lo or (extra and self.acc < pol.fb_hi)):
                self.st["prefetch_not_admitted"] += 1   # throttled by measured accuracy
                continue
            self._push(k, t, (s, lt), demand=False, p=float(P[e]))
            n += 1

    def _target_from(self, s, l):
        lt = l + self.pol.lookahead
        return (s, lt) if lt < self.L else (s + 1, lt - self.L)

    # ---------------------------------------------------------------- main loop
    def run(self):
        pol, L = self.pol, self.L
        t = 0.0
        per_step = []
        prefetching = pol.predictor is not None
        if prefetching and pol.lookahead >= 1:            # first d layers of step 0
            for lt in range(min(pol.lookahead, L)):
                self._issue(0, lt, pol.lookahead, 0.0)
        for s in range(self.steps):
            t0, stall = t, 0.0
            b0 = self.bytes_moved
            for l in range(L):
                tl0 = t
                if prefetching and pol.lookahead == 0:
                    self._issue(s, l, 0, t)
                t_route = t + self.nonexp[s]
                self._commit_until(t_route)
                t = t_route
                toks = {}
                for b in range(self.B):
                    for e in self.R[s, l, b]:
                        toks[int(e)] = toks.get(int(e), 0) + 1
                need = {(l, e) for e in toks}
                for k, r in list(self.queued.items()):     # promote / cancel queued prefetches
                    if r.target == (s, l) and not r.demand:
                        r.alive = False
                        self.queued.pop(k)
                        if k in need:
                            self._push(k, r.issue, (s, l), demand=True)
                        else:
                            self.st["prefetch_cancelled"] += 1
                            self._feedback(False)
                self.protect.pop((s, l), None)
                self.pred_p.pop((s, l), None)
                self.now = (s, l)
                for e, en in self.cache[l].items():     # score prefetches issued for this layer
                    if en.prefetched and en.target == (s, l) and not en.judged:
                        en.judged = True
                        hit = (l, e) in need
                        self.st["prefetch_on_target"] += int(hit)
                        self._feedback(hit)
                for k in need:
                    self.st["accesses"] += 1
                    if self.resident(k):
                        self.st["hits" if self.entry(k).ready <= t else "inflight_hits"] += 1
                    else:
                        self.st["misses"] += 1
                        if k not in self.queued:
                            self._push(k, t, (s, l), demand=True)
                self.pinned = set(need)
                if any(not self.resident(k) for k in need):
                    self._preempt(t, need)
                if prefetching and pol.lookahead >= 1:
                    self._issue(*self._target_from(s, l), pol.lookahead, t)
                while any(not self.resident(k) for k in need):
                    if not self._commit_next(float("inf")):
                        raise RuntimeError("demand fetch could not be scheduled")
                for k in sorted(need, key=lambda k: self.entry(k).ready):
                    en = self.entry(k)
                    if en.ready > t:
                        stall += en.ready - t
                        t = en.ready
                    t += costs.expert_exec_s(toks[k[1]], self.plat, self.model)
                    en.last_use = t
                    if en.prefetched and not en.used:
                        self.st["prefetch_used"] += 1
                    en.used = True
                self.pinned = set()
                if not pol.reuse:
                    for k in need:
                        self.cache[l].pop(k[1], None)
                else:
                    self._trim(l)
                dt = t - tl0
                self.layer_time = dt if self.layer_time is None else 0.8 * self.layer_time + 0.2 * dt
            self._commit_until(t)
            per_step.append((t - t0, stall, self.bytes_moved - b0))
        self.st["unused_at_end"] = sum(1 for part in self.cache for en in part.values()
                                       if en.prefetched and not en.used)
        return np.array(per_step), self.st


def summarize(per_step, st, plat, model, energy_rows=None):
    """Aggregate a run.  energy_rows: per-step simulator energies (J) of the work
    that does not depend on the policy (non-expert ops + expert execution)."""
    lat, stall, moved = per_step[:, 0], per_step[:, 1], per_step[:, 2]
    nbytes = moved.sum()
    n = len(lat)
    full = (st["demand_fetch"] + st["prefetch_fetch"]) * model.expert_bytes
    out = {
        "steps": n,
        "latency_ms_per_step": 1e3 * lat.mean(),
        "stall_ms_per_step": 1e3 * stall.mean(),
        "stall_frac": stall.sum() / lat.sum(),
        "traffic_gb_per_step": nbytes / n / 1e9,
        "demand_fetches": st["demand_fetch"], "prefetch_fetches": st["prefetch_fetch"],
        "hit_rate": st["hits"] / max(st["accesses"], 1),
        "inflight_hit_rate": st["inflight_hits"] / max(st["accesses"], 1),
        "miss_rate": st["misses"] / max(st["accesses"], 1),
        # accuracy = the prefetched expert was routed at the layer it was predicted for;
        # eventual use = it was executed at some point before eviction (no wasted traffic)
        "prefetch_accuracy": st["prefetch_on_target"] / max(st["prefetch_fetch"], 1),
        "prefetch_eventual_use": st["prefetch_used"] / max(st["prefetch_fetch"], 1),
        "prefetch_wasted": st["prefetch_fetch"] - st["prefetch_used"],
        "prefetch_dropped": st["prefetch_dropped"], "prefetch_cancelled": st["prefetch_cancelled"],
        "prefetch_not_admitted": st["prefetch_not_admitted"],
        "prefetch_aborted": st["prefetch_aborted"],
        "aborted_gb_per_step": (nbytes - full) / n / 1e9,          # partial copies cut by preemption
    }
    e_copy = costs.copy_energy_j(nbytes, plat)
    e_bg = plat.background_w * lat.sum()
    e_fixed = float(np.sum(energy_rows)) if energy_rows is not None else 0.0
    out.update({
        "energy_copy_j_per_step": e_copy / n,
        "energy_background_j_per_step": e_bg / n,
        "energy_stall_background_j_per_step": plat.background_w * stall.sum() / n,
        "energy_fixed_j_per_step": e_fixed / n,
        "energy_total_j_per_step": (e_copy + e_bg + e_fixed) / n,
    })
    return out
