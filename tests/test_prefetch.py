"""Unit tests for the predictors and the timeline simulator (run: python3 -m pytest -q tests
or python3 tests/test_prefetch.py).  Pure numpy, no simulator build needed."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prefetch import costs, predictors as P  # noqa: E402
from prefetch.timeline import Sim, Policy, summarize  # noqa: E402
from prefetch.workload import write_sim_trace, read_sim_log  # noqa: E402

PLAT = costs.H100_NVL5
LOAD = costs.expert_load_s(PLAT)
X1 = costs.expert_exec_s(1, PLAT)
ATT = 30e-6


def make_R(steps, L, B, fn):
    R = np.zeros((steps, L, B, 2), dtype=np.int64)
    for s in range(steps):
        for l in range(L):
            for b in range(B):
                R[s, l, b] = fn(s, l, b)
    return R


def H_from_R(R, prompt=1):
    steps, L, B, K = R.shape
    H = np.zeros((B, prompt + steps, L, K), dtype=np.int64)
    H[:, prompt:] = R.transpose(2, 0, 1, 3)
    H[:, :prompt] = R[0].transpose(1, 0, 2)[:, None]
    return H


def run(R, pol, prompt=1, att=ATT):
    H = H_from_R(R, prompt)
    nonexp = np.full(R.shape[0], att)
    return Sim(R, H, prompt, nonexp, PLAT, pol).run()


def test_on_demand_two_misses_closed_form():
    """1 step, 1 layer, two misses: e1 arrives at t+LOAD, e2 at t+2*LOAD."""
    R = make_R(1, 1, 1, lambda s, l, b: [0, 1])
    pol = Policy(capacity=8, reuse=False)
    H = H_from_R(R)
    per_step, st = Sim(R, H, 1, np.array([ATT]), PLAT, pol).run()
    lat, stall = per_step[0, 0], per_step[0, 1]
    assert np.isclose(lat, ATT + 2 * LOAD + X1)
    assert np.isclose(stall, LOAD + (2 * LOAD - (LOAD + X1)))
    assert st["demand_fetch"] == 2 and st["prefetch_fetch"] == 0


def test_framework_no_reuse_matches_copy_bytes():
    rng = np.random.default_rng(0)
    R = make_R(5, 32, 4, lambda s, l, b: rng.choice(8, 2, replace=False))
    per_step, st = run(R, Policy(capacity=64, reuse=False))
    distinct = sum(len(np.unique(R[s, l])) for s in range(5) for l in range(32))
    assert st["demand_fetch"] == distinct
    assert np.isclose(per_step[:, 2].sum(), distinct * costs.MIXTRAL.expert_bytes)


def test_cache_reuse_removes_stalls():
    """Same experts every step and enough capacity: only step 0 stalls."""
    R = make_R(4, 32, 1, lambda s, l, b: [l % 8, (l + 1) % 8])
    per_step, st = run(R, Policy(capacity=64, reuse=True))
    assert per_step[0, 1] > 0
    assert np.allclose(per_step[1:, 1], 0)
    assert st["demand_fetch"] == 64


def test_oracle_prefetch_hides_load_when_compute_is_long():
    """If each layer computes longer than it takes to copy the next layer's experts,
    an oracle prefetcher leaves only the first layer's stall."""
    R = make_R(1, 4, 1, lambda s, l, b: [2 * l % 8, (2 * l + 1) % 8])
    long_att = 3 * LOAD
    _, st0 = run(R, Policy(capacity=32), att=long_att)
    per_step, st = run(R, Policy("o", 32, True, P.Oracle(), 1, ("set",)), att=long_att)
    assert np.isclose(per_step[0, 1], 0, atol=1e-12), per_step
    assert st["prefetch_used"] == 8 - 2 or st["prefetch_used"] == 8


def test_demand_overtakes_queued_prefetch_and_preemption_bounds_delay():
    """A wrong in-flight prefetch delays a demand by at most one chunk with preempt."""
    class Wrong(P.Predictor):
        def probs(self, ctx, s, lt, b, d):
            p = np.zeros(8)
            p[[6, 7]] = 1.0                                # never routed
            return p
    R = make_R(1, 2, 1, lambda s, l, b: [0, 1])
    base, _ = run(R, Policy(capacity=32), att=1e-6)
    slow, _ = run(R, Policy("w", 32, True, Wrong(), 1, ("fixed", 2)), att=1e-6)
    pre, st = run(R, Policy("w", 32, True, Wrong(), 1, ("fixed", 2), preempt=True), att=1e-6)
    assert slow[0, 0] >= base[0, 0]
    assert pre[0, 0] <= slow[0, 0] + 1e-12
    assert pre[0, 0] - base[0, 0] <= 2 * LOAD / 3 + 1e-9
    assert st["prefetch_aborted"] >= 1


def test_predictors_are_causal():
    """Changing routing that is not yet known at issue time must not change a prediction."""
    rng = np.random.default_rng(1)
    prof = np.argsort(rng.random((6, 40, 32, 8)), -1)[..., :2]
    tab = P.Tables(prof)
    lam = {1: 0.5, 2: 0.5}
    H = np.argsort(rng.random((2, 20, 32, 8)), -1)[..., :2]
    tok0 = 4
    for pred in (P.Temporal(tab), P.Spatial(tab), P.STHybrid(tab, lam), P.Frequency(tab)):
        for d in (1, 2):
            for s, lt in ((3, 0), (3, 1), (5, 17), (7, 31)):
                ctx = P.Context(H.copy(), tok0)
                p0 = pred.probs(ctx, s, lt, 0, d)
                H2 = H.copy()
                t = tok0 + s
                H2[:, t + 1:] = (H2[:, t + 1:] + 3) % 8                        # future tokens
                ls = lt - d
                if ls >= 0:
                    H2[:, t, ls + 1:] = (H2[:, t, ls + 1:] + 5) % 8              # not-yet-routed layers
                else:
                    H2[:, t] = (H2[:, t] + 5) % 8
                    H2[:, t - 1, 32 + ls + 1:] = (H2[:, t - 1, 32 + ls + 1:] + 5) % 8
                p1 = pred.probs(P.Context(H2, tok0), s, lt, 0, d)
                assert np.allclose(p0, p1), (pred.name, d, s, lt)


def test_stateful_and_speculative_predictors_are_causal():
    """ST-MoE (d=1, called in time order) and the speculative predictors read only
    routing / router outputs that exist when the prediction is issued."""
    rng = np.random.default_rng(3)
    prof = np.argsort(rng.random((6, 40, 32, 8)), -1)[..., :2]
    tab = P.Tables(prof)
    H = np.argsort(rng.random((1, 12, 32, 8)), -1)[..., :2]
    spec = {1: rng.random((1, 12, 32, 8)).astype(np.float32)}
    tok0 = 4
    targets = [(s, lt) for s in range(6) for lt in range(32)]

    def run_all(H, spec, make):
        pred = make()
        ctx = P.Context(H, tok0, spec)
        return [pred.probs(ctx, s, lt, 0, 1) for s, lt in targets]

    for make in (lambda: P.STMoE(tab), lambda: P.SpecGate(), lambda: P.SpecHybrid(tab, {1: 0.5, 2: 0.5})):
        ref = run_all(H, spec, make)
        for i, (s, lt) in enumerate(targets[::17]):
            j = targets.index((s, lt))
            H2, sp2 = H.copy(), {1: spec[1].copy()}
            t = tok0 + s
            H2[:, t, lt:] = (H2[:, t, lt:] + 3) % 8            # target layer and later: unknown
            H2[:, t + 1:] = (H2[:, t + 1:] + 5) % 8
            sp2[1][:, t, lt + 1:] = rng.random(sp2[1][:, t, lt + 1:].shape)
            sp2[1][:, t + 1:] = rng.random(sp2[1][:, t + 1:].shape)
            got = run_all(H2, sp2, make)
            a, b = ref[j], got[j]
            assert (a is None and b is None) or np.allclose(a, b), (make().name, s, lt)


def test_stmoe_rejects_other_lookahead():
    rng = np.random.default_rng(4)
    prof = np.argsort(rng.random((4, 20, 32, 8)), -1)[..., :2]
    pred = P.STMoE(P.Tables(prof))
    try:
        pred.probs(P.Context(prof[:1], 2), 0, 5, 0, 2)
    except ValueError:
        return
    raise AssertionError("ST-MoE must refuse d != 1")


def test_accuracy_counts_only_the_target_layer():
    """A prefetch that is wrong for its target but used a step later is 'eventually used',
    not 'accurate'."""
    class WrongNow(P.Predictor):
        def probs(self, ctx, s, lt, b, d):
            if (s, lt) != (0, 1):
                return None                                # no prediction elsewhere
            p = np.zeros(8)
            p[[4, 5]] = 1.0                                # routed only at step 1
            return p
    R = make_R(2, 2, 1, lambda s, l, b: [4, 5] if (s == 1 and l == 1) else [2 * l, 2 * l + 1])
    _, st = run(R, Policy("w", 32, True, WrongNow(), 1, ("fixed", 2)), att=3 * LOAD)
    assert st["prefetch_fetch"] == 2 and st["prefetch_on_target"] == 0 and st["prefetch_used"] == 2


def test_preempt_keeps_likely_prefetch_for_later_layer():
    """With d=2 a demand miss at layer 0 must not abort a confident prefetch for layer 1."""
    class Pred(P.Predictor):
        def probs(self, ctx, s, lt, b, d):
            if lt != 1:
                return None                                # layer 0 not predicted
            p = np.zeros(8)
            p[[2, 3]] = 0.95                               # layer 1 right
            return p
    R = make_R(1, 3, 1, lambda s, l, b: [2 * l, 2 * l + 1])
    _, st = run(R, Policy("p", 32, True, Pred(), 2, ("fixed", 2), preempt=True), att=1e-6)
    assert st["prefetch_aborted"] == 0 and st["prefetch_on_target"] == 2


def test_adaptive_budget():
    p = np.array([0.9, 0.8, 0.2, 0.05, 0.05, 0, 0, 0])
    assert len(P.select(p, 2, ("adaptive", 0.8, 6))) == 2
    assert len(P.select(p, 2, ("adaptive", 0.95, 6))) == 3
    flat = np.full(8, 0.25)
    assert len(P.select(flat, 2, ("adaptive", 0.9, 4))) == 4


def test_sim_trace_roundtrip(tmp_path="/tmp"):
    rng = np.random.default_rng(2)
    R = np.argsort(rng.random((3, 32, 4, 8)), -1)[..., :2]
    f = os.path.join(tmp_path, "rt_test.txt")
    write_sim_trace(R, f)
    with open(f) as fh:
        B, steps, L, K = map(int, fh.readline().split())
        arr = np.array(fh.read().split(), dtype=np.int64).reshape(B, steps, L, K)
    assert (arr.transpose(1, 2, 0, 3) == R).all()
    log = os.path.join(tmp_path, "rl_test.csv")
    with open(log, "w") as fh:
        fh.write("call,step,layer,slot,e0,e1\n")
        for s in range(3):
            for l in range(32):
                for b in range(4):
                    fh.write(f"{s * 32 + l},{s},{l},{b},{R[s, l, b, 0]},{R[s, l, b, 1]}\n")
    assert (read_sim_log(log, 32, 2) == R).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
