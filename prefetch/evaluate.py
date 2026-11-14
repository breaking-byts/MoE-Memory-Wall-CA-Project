"""Offline prediction-accuracy evaluation against the actually routed experts.

Metrics (per decode token, per target layer; batch-level = union over the batch):
  recall / overlap accuracy  |P ∩ A| / |A|     fraction of routed experts that were predicted
  precision                  |P ∩ A| / |P|     fraction of prefetched experts that get used
  full coverage              1[A ⊆ P]          APEX's stall-relevant criterion: no demand miss
  set size                   |P|               prefetch budget actually spent
"""
import numpy as np

from .predictors import Context, select


def evaluate(pred, wl, d, policy, slots=None, batch=1, max_steps=None):
    """Run predictor `pred` with lookahead d over decode steps of workload `wl`.

    Sequences are grouped into batches of `batch` slots that decode in lock-step;
    for batch > 1 the prediction is the union of per-slot predictions and the
    reference is the union of the routed experts (what must be in HBM)."""
    slots = list(range(wl.S)) if slots is None else list(slots)
    steps = min(wl.decode_steps, max_steps or 10 ** 9)
    spec = {k: v for k, v in wl.spec.items()}
    rec, prec, cov, size, lay = [], [], [], [], []
    for g in range(0, len(slots) - batch + 1, batch):
        group = slots[g:g + batch]
        if hasattr(pred, "flush"):
            pred.flush()
        ctx = Context(wl.ids[group], wl.prompt_len,
                      {k: v[group] for k, v in spec.items()}, wl.E)
        for s in range(steps):
            for lt in range(wl.L):
                P, A = set(), set()
                for b in range(batch):
                    p = pred.probs(ctx, s, lt, b, d)
                    P.update(select(p, wl.K, policy).tolist())
                    A.update(ctx.H[b, ctx.tok0 + s, lt].tolist())
                hit = len(P & A)
                rec.append(hit / len(A))
                lay.append(lt)
                prec.append(hit / len(P) if P else np.nan)
                cov.append(float(A <= P))
                size.append(len(P))
    rec, lay = np.array(rec), np.array(lay)
    return {"recall": float(np.mean(rec)), "precision": float(np.nanmean(prec)),
            "coverage": float(np.mean(cov)), "size": float(np.mean(size)),
            "n": len(rec),
            "recall_per_layer": np.array([rec[lay == l].mean() for l in range(wl.L)])}


def correlation_stats(ids, E=8):
    """Temporal / spatial locality statistics of a routing trace [S, T, L, K]."""
    S, T, L, K = ids.shape
    oh = np.zeros((S, T, L, E), dtype=np.float32)
    np.put_along_axis(oh, ids, 1.0, axis=-1)
    tok = (oh[:, 1:] * oh[:, :-1]).sum(-1) / K                      # S,T-1,L overlap fraction
    lay = (oh[:, :, 1:] * oh[:, :, :-1]).sum(-1) / K                # same-id overlap l -> l+1
    rep1 = (ids[:, 1:, :, 0] == ids[:, :-1, :, 0]).mean((0, 1))      # Mixtral Table 5 "first choice"
    rep12 = ((oh[:, 1:] * oh[:, :-1]).sum(-1) >= 1).mean((0, 1))    # "first or second choice"
    freq = oh.mean((0, 1))                                           # L,E
    # transition-matrix predictability: mutual information between expert sets of l and l+1
    trans = np.einsum("stle,stlf->lef", oh[:, :, :-1], oh[:, :, 1:])
    return {"token_overlap": float(tok.mean()), "token_overlap_per_layer": tok.mean((0, 1)),
            "layer_overlap_same_id": float(lay.mean()),
            "rep_first": rep1, "rep_first_or_second": rep12,
            "freq": freq, "trans": trans}
