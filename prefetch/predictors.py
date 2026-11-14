"""Training-free expert predictors.

All predictors are causal: a prediction for (decode step s, target layer lt) made
with lookahead d may only use information available when layer lt-d has been
routed (for lt-d < 0: layer L+lt-d of the previous step).  Nothing is trained
with gradients.  Two predictors use *count tables* built once from a disjoint
profiling split (this is how ST-MoE builds its Cross-layer Correlation Table);
one (speculative gating) re-uses the model's own router weights.

Interface
---------
  p = pred.probs(ctx, s, lt, b, d)  -> np.ndarray[E], expected selection probability
                                       of each expert (sums to ~K), or None
  ctx: Context holding the per-slot routing history of the running batch.

Budget policies (which experts to prefetch from p) live in `select()`.
"""
import numpy as np


class Context:
    """Routing history visible to the predictor.

    H[b, t, l, :] = experts of slot b at absolute token t, layer l (prompt + decode).
    tok0 = absolute token index of decode step 0 (= prompt length).
    spec[d][b, t, l, :] = speculative router probabilities (if the workload has them)."""

    def __init__(self, H, tok0, spec=None, E=8):
        self.H, self.tok0, self.spec, self.E = H, tok0, spec or {}, E
        self.B, self.T, self.L, self.K = H.shape

    def source(self, s, lt, d):
        """(token, layer) whose routing is the latest one known at issue time."""
        t = self.tok0 + s
        ls = lt - d
        if ls >= 0:
            return t, ls
        return t - 1, ls + self.L


# ----------------------------------------------------------------------------- tables
def onehot_sets(ids, E):
    """ids [..., K] -> {0,1} [..., E]"""
    oh = np.zeros(ids.shape[:-1] + (E,), dtype=np.float32)
    np.put_along_axis(oh, ids, 1.0, axis=-1)
    return oh


class Tables:
    """Count tables from the profiling split (S', T, L, K)."""

    def __init__(self, prof, E=8, max_d=3, alpha=0.5):
        S, T, L, K = prof.shape
        self.E, self.L, self.K = E, L, K
        oh = onehot_sets(prof, E)                                 # S,T,L,E
        self.freq = oh.mean((0, 1))                               # L,E  P(e at l)
        # spatial: M[d][l][a, e] = P(e in A_l | a in A_{l-d}) for the same token;
        # for l-d < 0 the source is layer L+l-d of the previous token.
        flat = oh.reshape(S * T, L, E)
        self.M = {}
        for d in range(1, max_d + 1):
            Md = np.zeros((L, E, E), dtype=np.float64)
            for l in range(L):
                if l - d >= 0:
                    src, dst = flat[:, l - d], flat[:, l]
                else:                                             # wrap to previous token
                    src = oh[:, :-1, L + l - d].reshape(-1, E)
                    dst = oh[:, 1:, l].reshape(-1, E)
                c = src.T @ dst + alpha * self.freq[l][None, :]   # smoothed co-occurrence
                Md[l] = c / (src.sum(0)[:, None] + alpha)
            self.M[d] = Md
        # temporal: P(e in A_{t,l} | e in A_{t-1,l}) and P(e in A_{t,l} | e not in A_{t-1,l})
        prev, cur = oh[:, :-1], oh[:, 1:]
        stay = (prev * cur).sum((0, 1)) / np.maximum(prev.sum((0, 1)), 1)       # L,E
        enter = ((1 - prev) * cur).sum((0, 1)) / np.maximum((1 - prev).sum((0, 1)), 1)
        self.stay, self.enter = stay, enter


# ----------------------------------------------------------------------------- predictors
class Predictor:
    name = "base"
    needs_spec = False

    def probs(self, ctx, s, lt, b, d):
        raise NotImplementedError


class Oracle(Predictor):
    name = "oracle"

    def probs(self, ctx, s, lt, b, d):
        return onehot_sets(ctx.H[b, ctx.tok0 + s, lt], ctx.E)


class Frequency(Predictor):
    """Static popularity per layer (profile)."""
    name = "frequency"

    def __init__(self, tab):
        self.tab = tab

    def probs(self, ctx, s, lt, b, d):
        return self.tab.freq[lt] * ctx.K


class Temporal(Predictor):
    """History Table: the same layer's experts for the previous token of this sequence."""
    name = "temporal"

    def __init__(self, tab):
        self.tab = tab

    def probs(self, ctx, s, lt, b, d):
        t = ctx.tok0 + s
        prev = onehot_sets(ctx.H[b, t - 1, lt], ctx.E)
        return prev * self.tab.stay[lt] + (1 - prev) * self.tab.enter[lt]


class Spatial(Predictor):
    """Cross-layer Correlation Table: P(e at lt | experts at the source layer)."""
    name = "spatial"

    def __init__(self, tab):
        self.tab = tab

    def probs(self, ctx, s, lt, b, d):
        d = max(d, 1)
        t, ls = ctx.source(s, lt, d)
        src = ctx.H[b, t, ls]
        return self.tab.M[d][lt][src].mean(0)


class STHybrid(Predictor):
    """Spatial + temporal evidence, mixed per lookahead with weight lam.

    lam is chosen on the profiling split (grid search, no gradients)."""
    name = "st-hybrid"

    def __init__(self, tab, lam=None):
        self.sp, self.tp, self.lam = Spatial(tab), Temporal(tab), lam or {}

    def probs(self, ctx, s, lt, b, d):
        lam = self.lam.get(max(d, 1), 0.5)
        return lam * self.sp.probs(ctx, s, lt, b, d) + (1 - lam) * self.tp.probs(ctx, s, lt, b, d)


class STMoE(Predictor):
    """Re-implementation of the ST-MoE Expert Prediction Unit (Zhao et al., 2026).

    CCT: per (layer, expert) the top-K co-activated experts of the next layer, each
    with a 2-bit confidence counter starting at 2 (0b10), +1 on a correct prediction
    (max 3), -1 otherwise; an entry whose counter reaches 0 is replaced by the next
    most co-activated expert.  HT: last token's experts per layer, fixed score 2.
    An expert is predicted iff (CCT score + HT score) >= 2, so the set size varies.
    Counters are updated online from the routing actually observed.
    probs() returns score / 8 (max score 3+3+2), so the ST-MoE rule is the budget
    ("min_p", 0.25) and the scores can rank residents for replacement.  Defined for
    the next layer only (lookahead d = 1)."""
    name = "st-moe"

    def __init__(self, tab):
        self.tab = tab
        L, E, K = tab.L, tab.E, tab.K
        self.rank = np.argsort(-tab.M[1], axis=-1)                    # L, E(src), E ordered dst
        self.entries = self.rank[..., :K].copy()                       # current CCT entries
        self.next_alt = np.full((L, E), K, dtype=np.int64)
        self.score = np.full((L, E, K), 2, dtype=np.int64)
        self.pending = []          # counter updates become visible once the target layer ran

    def probs(self, ctx, s, lt, b, d):
        if d != 1:
            raise ValueError("ST-MoE's correlation table predicts the next layer only (d = 1)")
        key = (s, lt)
        keep = []
        for k, upd in self.pending:
            if k < key:
                self._update(*upd)
            else:
                keep.append((k, upd))
        self.pending = keep
        t, ls = ctx.source(s, lt, 1)
        p = np.zeros(ctx.E)
        for a in ctx.H[b, t, ls]:
            for j, e in enumerate(self.entries[lt, a]):
                p[e] += self.score[lt, a, j]
        for e in ctx.H[b, ctx.tok0 + s - 1, lt]:                       # history table
            p[e] += 2
        self.pending.append((key, (lt, ctx.H[b, t, ls], ctx.H[b, ctx.tok0 + s, lt])))
        return p / 8.0                                                   # predicted iff >= 2/8

    def flush(self):
        """Apply all outstanding counter updates (call between independent batches)."""
        for _, upd in self.pending:
            self._update(*upd)
        self.pending = []

    def _update(self, lt, srcs, actual):
        act = set(actual.tolist())
        for a in srcs:
            for j in range(self.tab.K):
                e = self.entries[lt, a, j]
                if e in act:
                    self.score[lt, a, j] = min(3, self.score[lt, a, j] + 1)
                else:
                    self.score[lt, a, j] -= 1
                    if self.score[lt, a, j] <= 0:                        # replace entry
                        cand = self.rank[lt, a]
                        k = self.next_alt[lt, a]
                        while cand[k % self.tab.E] in self.entries[lt, a]:
                            k += 1
                        self.entries[lt, a, j] = cand[k % self.tab.E]
                        self.next_alt[lt, a] = k + 1
                        self.score[lt, a, j] = 2


NBINS = 20


def spec_calibration(prof_ids, prof_spec, d, nbins=NBINS, alpha=1.0):
    """Histogram calibration of speculative router outputs (profiling split only).

    A router softmax value is not the probability that the expert ends up in the
    top-K.  Table[r, j] = P(expert is routed | it is ranked r-th by the speculative
    router and its softmax value falls in bin j), counted on the profiling split
    (Laplace-smoothed towards the rank's marginal).  No gradients, no parameters
    beyond these counts."""
    S, T, L, K = prof_ids.shape
    E = prof_spec.shape[-1]
    q = prof_spec[:, :, d:].reshape(-1, E)                    # layers >= d have a prediction
    sel = onehot_sets(prof_ids[:, :, d:], E).reshape(-1, E)
    rank = np.argsort(np.argsort(-q, axis=1, kind="stable"), axis=1)
    b = np.minimum((q * nbins).astype(int), nbins - 1)
    hit = np.zeros((E, nbins))
    cnt = np.zeros((E, nbins))
    np.add.at(hit, (rank.ravel(), b.ravel()), sel.ravel())
    np.add.at(cnt, (rank.ravel(), b.ravel()), 1)
    marg = hit.sum(1, keepdims=True) / np.maximum(cnt.sum(1, keepdims=True), 1)
    return (hit + alpha * marg) / (cnt + alpha)


class SpecGate(Predictor):
    """Speculative gating: the target layer's own router applied to the hidden
    state available d layers earlier (training-free, re-uses router weights).
    With `calib` (from spec_calibration) the router outputs are mapped to
    P(routed); without it, softmax x K is used."""
    name = "spec-gate"
    needs_spec = True

    def __init__(self, calib=None):
        self.calib = calib or {}

    def probs(self, ctx, s, lt, b, d):
        if d not in ctx.spec or lt - d < 0:
            return None
        q = ctx.spec[d][b, ctx.tok0 + s, lt]
        if d not in self.calib:
            return q * ctx.K
        rank = np.argsort(np.argsort(-q, kind="stable"), kind="stable")
        return self.calib[d][rank, np.minimum((q * NBINS).astype(int), NBINS - 1)]


class SpecHybrid(Predictor):
    """Calibrated speculative gate where an earlier hidden state exists; for the
    first d layers of a token (no earlier hidden state of that token) it falls back
    to the history-based ST-hybrid prediction."""
    name = "spec-hybrid"
    needs_spec = True

    def __init__(self, tab, lam=None, calib=None):
        self.sg, self.st = SpecGate(calib), STHybrid(tab, lam)

    def probs(self, ctx, s, lt, b, d):
        p = self.sg.probs(ctx, s, lt, b, d)
        return self.st.probs(ctx, s, lt, b, d) if p is None else p


class RandomPred(Predictor):
    name = "random"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def probs(self, ctx, s, lt, b, d):
        return self.rng.random(ctx.E)


# ----------------------------------------------------------------------------- budgets
def select(p, K, policy):
    """Choose the experts to prefetch from probabilities p.

    policy:
      ("fixed", n)        top-n experts
      ("adaptive", tau, nmax)
            APEX-inspired, training-free: smallest n >= K whose cumulative expected
            coverage sum_{top-n} p_e / K reaches tau (capped at nmax).  Confident
            predictions -> few experts, uncertain -> more, instead of a fixed budget.
      ("set",)            predictor already returns a 0/1 set (oracle)
      ("min_p", theta)    every expert with p >= theta (ST-MoE: score >= 2 <=> p >= 0.25)
    """
    if p is None:
        return np.array([], dtype=np.int64)
    if policy[0] == "set":
        return np.flatnonzero(p > 0.5)
    if policy[0] == "min_p":
        return np.flatnonzero(p >= policy[1] - 1e-12)
    order = np.argsort(-p, kind="stable")
    if policy[0] == "fixed":
        return order[:policy[1]]
    tau, nmax = policy[1], policy[2]
    cum = np.cumsum(p[order]) / K
    n = int(np.searchsorted(cum, tau - 1e-9) + 1)
    return order[:min(max(n, K), nmax)]


def tune_lambda(tab, prof, max_d=2, grid=np.linspace(0, 1, 11)):
    """Pick the spatial/temporal mixing weight per lookahead on the profiling split
    (maximises top-K recall).  Only count tables are involved."""
    S, T, L, K = prof.shape
    ctx = Context(prof, tok0=1)
    sp, tp = Spatial(tab), Temporal(tab)
    rng = np.random.default_rng(1)
    samples = [(rng.integers(0, S), rng.integers(0, T - 2), rng.integers(0, L)) for _ in range(3000)]
    lam = {}
    for d in range(1, max_d + 1):
        best, arg = -1, 0.5
        P_sp = [sp.probs(ctx, s, l, b, d) for b, s, l in samples]
        P_tp = [tp.probs(ctx, s, l, b, d) for b, s, l in samples]
        act = [set(prof[b, 1 + s, l]) for b, s, l in samples]
        for g in grid:
            r = np.mean([len(set(np.argsort(-(g * a + (1 - g) * c))[:K]) & A) / K
                         for a, c, A in zip(P_sp, P_tp, act)])
            if r > best:
                best, arg = r, float(g)
        lam[d] = arg
    return lam
