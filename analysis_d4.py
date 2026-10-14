"""D4: temporal / spatial correlation of expert accesses (Mixtral: 8 experts, top-2, 32 layers).

LLMSimulator does not ship real routing traces. Its own routing is either uniform
(getRandomExpert) or Zipf-skewed (getZipfianRandomExpert, w ~ 1/(k+1)^s), so we
analyse those two modes plus one ST-MoE-like correlated trace. The correlated
trace is SYNTHETIC: its correlation is set by STAY / PERSIST below, so its
numbers demonstrate the method, they are not a measurement of real routing.

Output: d4_correlation.png and the statistics table below.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

E, K, L = 8, 2, 32                     # experts, top-k, MoE layers
SEQ, TOK, BATCH = 64, 64, 8            # sequences x consecutive tokens; decode batch
ZIPF_S, STAY, PERSIST = 1.0, 0.6, 0.7  # skew; P(keep expert at next layer); P(keep at next token)
rng = np.random.default_rng(0)


def zipf(s):
    w = 1.0 / np.arange(1, E + 1) ** s
    return w / w.sum()


def draw(p):
    return rng.choice(E, size=K, replace=False, p=p)


def trace(skew, correlated):
    """Return routed experts, shape (SEQ, TOK, L, K)."""
    p = zipf(ZIPF_S if skew else 0.0)
    out = np.zeros((SEQ, TOK, L, K), dtype=int)
    for s in range(SEQ):
        for t in range(TOK):
            for l in range(L):
                if correlated and t > 0 and rng.random() < PERSIST:
                    cur = out[s, t - 1, l]                      # same experts as previous token
                elif correlated and l > 0:
                    prev = out[s, t, l - 1]                     # walk from the previous layer
                    nxt = [c if rng.random() < STAY else rng.choice(E, p=p) for c in prev]
                    cur = np.array(nxt) if nxt[0] != nxt[1] else draw(p)
                else:
                    cur = draw(p)
                out[s, t, l] = cur
    return out


def stats(tr):
    hits = lambda a, b: (a[..., :, None] == b[..., None, :]).any(-1).mean()
    trans = np.zeros((E, E))
    for a in range(K):
        for b in range(K):
            np.add.at(trans, (tr[:, :, :-1, a].ravel(), tr[:, :, 1:, b].ravel()), 1)
    picks = [rng.choice(SEQ, BATCH, replace=False) for _ in range(2000)]  # 8 requests, same step
    tt, ll = rng.integers(0, TOK, 2000), rng.integers(0, L, 2000)
    cover = np.mean([len(set(tr[p, t, l].ravel())) for p, t, l in zip(picks, tt, ll)])
    return {
        "freq": np.bincount(tr.ravel(), minlength=E) / tr.size,
        "heat": np.stack([np.bincount(tr[:, :, l].ravel(), minlength=E) for l in range(L)]),
        "trans": trans / trans.sum(1, keepdims=True),
        "layer_overlap": hits(tr[:, :, 1:], tr[:, :, :-1]),   # spatial: layer l -> l+1
        "token_overlap": hits(tr[:, 1:], tr[:, :-1]),         # temporal: token t -> t+1
        "coverage": cover,
    }


scenarios = {
    "uniform (sim default)": trace(skew=False, correlated=False),
    f"Zipf s={ZIPF_S} (sim skewness)": trace(skew=True, correlated=False),
    "ST-MoE-like (synthetic)": trace(skew=True, correlated=True),
}
print(f"random-guess overlap null = K/E = {K / E:.2f}")
print(f"{'scenario':<28s} {'top share':>9s} {'layer ovl':>10s} {'token ovl':>10s} "
      f"{'experts/layer @ batch 8':>24s}")
fig, ax = plt.subplots(len(scenarios), 3, figsize=(14, 11), constrained_layout=True)
for i, (name, tr) in enumerate(scenarios.items()):
    s = stats(tr)
    print(f"{name:<28s} {s['freq'].max():9.3f} {s['layer_overlap']:10.3f} "
          f"{s['token_overlap']:10.3f} {s['coverage']:24.2f}")
    ax[i, 0].bar(range(E), s["freq"]); ax[i, 0].set_title(f"{name}\nexpert frequency")
    ax[i, 1].imshow(s["heat"], aspect="auto"); ax[i, 1].set_title("layer x expert counts")
    im = ax[i, 2].imshow(s["trans"], vmin=0, vmax=0.6); ax[i, 2].set_title("P(expert at l+1 | expert at l)")
    ax[i, 0].set_xlabel("expert"); ax[i, 1].set_xlabel("expert"); ax[i, 1].set_ylabel("layer")
    ax[i, 2].set_xlabel("expert at l+1"); ax[i, 2].set_ylabel("expert at l")
fig.colorbar(im, ax=ax[:, 2])
fig.savefig("d4_correlation.png", dpi=110)
print("wrote d4_correlation.png")
