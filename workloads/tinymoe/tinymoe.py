"""TinyMoE: a small Mixtral-topology MoE language model (32 layers, 8 experts, top-2).

Why this exists
---------------
LLMSimulator / SSD-offloading ship no real routing traces (their routing is a
uniform or Zipf random draw, see midterm D4), and real Mixtral-8x7B weights cannot
be run in this project's CPU-only environment. To evaluate an expert predictor
against *learned* router decisions (instead of hand-set correlation knobs), we
train a small model with exactly Mixtral's MoE topology -- 32 MoE layers, 8
experts per layer, top-2 softmax-renormalised gating, SwiGLU experts, Switch-style
load-balancing loss -- on real text (prose + Python code) and record its routing.

Only the routing *pattern* (which expert ids are chosen per token per layer) is
taken from TinyMoE. Latency / energy always use Mixtral-8x7B expert sizes from the
SSD-offloading simulator (352.3 MB per expert).

This file holds the model, tokenizer and data helpers shared by train/extract.
"""
import os
import json
import numpy as np
import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(HERE, "corpus")
CKPT = os.path.join(HERE, "checkpoint")

CFG = dict(
    vocab=4096,      # byte-level BPE trained on both domains
    ctx=256,         # tokens per sequence
    d=64,            # model width
    heads=4,
    layers=32,       # = Mixtral
    experts=8,       # = Mixtral
    top_k=2,         # = Mixtral
    ffn=128,         # expert SwiGLU hidden width (2 x d)
    aux_coef=0.01,   # Switch-Transformer load-balancing loss coefficient
)


# ----------------------------------------------------------------------------- data
def load_tokenizer():
    from tokenizers import ByteLevelBPETokenizer
    tok_dir = os.path.join(CKPT, "tokenizer")
    vocab, merges = os.path.join(tok_dir, "vocab.json"), os.path.join(tok_dir, "merges.txt")
    if os.path.exists(vocab):
        return ByteLevelBPETokenizer(vocab, merges)
    os.makedirs(tok_dir, exist_ok=True)
    tok = ByteLevelBPETokenizer()
    tok.train([os.path.join(CORPUS, f) for f in ("prose.txt", "code.txt")],
              vocab_size=CFG["vocab"], min_frequency=2, show_progress=False)
    tok.save_model(tok_dir)
    return ByteLevelBPETokenizer(vocab, merges)


def domain_splits(tok):
    """Per domain: train = first 90 % of characters, held-out = last 10 %.
    The held-out text is split again: first half = profiling set (used only to
    count co-occurrences for the ST-MoE table), second half = evaluation set."""
    out = {}
    for dom in ("prose", "code"):
        text = open(os.path.join(CORPUS, f"{dom}.txt"), encoding="utf-8").read()
        cut = int(len(text) * 0.9)
        held = text[cut:]
        half = len(held) // 2
        parts = {"train": text[:cut], "profile": held[:half], "eval": held[half:]}
        out[dom] = {k: np.array(tok.encode(v).ids, dtype=np.int32) for k, v in parts.items()}
    return out


def chunk(ids, ctx, n=None, stride=None):
    """Non-overlapping windows of length ctx (first n of them)."""
    stride = stride or ctx
    starts = list(range(0, len(ids) - ctx, stride))
    if n is not None:
        starts = starts[:n]
    return np.stack([ids[s:s + ctx] for s in starts])


# ----------------------------------------------------------------------------- model
def init_params(key, cfg=CFG):
    d, f, E, L, V = cfg["d"], cfg["ffn"], cfg["experts"], cfg["layers"], cfg["vocab"]
    ks = jax.random.split(key, 9)
    n = lambda k, shape, s: (jax.random.normal(k, shape) * s).astype(jnp.float32)
    res = 0.02 / np.sqrt(2 * L)            # GPT-2 style scaled residual init
    return {
        "emb": n(ks[0], (V, d), 0.02),
        "pos": n(ks[1], (cfg["ctx"], d), 0.01),
        "lnf": jnp.ones((d,)),
        "layers": {                         # stacked over layers -> lax.scan
            "ln1": jnp.ones((L, d)),
            "ln2": jnp.ones((L, d)),
            "wqkv": n(ks[2], (L, d, 3 * d), 0.02),
            "wo": n(ks[3], (L, d, d), res),
            "router": n(ks[4], (L, d, E), 0.02),
            "w1": n(ks[5], (L, E, d, f), 0.02),
            "w3": n(ks[6], (L, E, d, f), 0.02),
            "w2": n(ks[7], (L, E, f, d), res),
        },
    }


def rms(x, g):
    return x * jax.lax.rsqrt(jnp.mean(x * x, -1, keepdims=True) + 1e-6) * g


def attention(x, wqkv, wo, heads):
    B, T, d = x.shape
    q, k, v = jnp.split(x @ wqkv, 3, axis=-1)
    sh = lambda t: t.reshape(B, T, heads, d // heads).transpose(0, 2, 1, 3)
    q, k, v = sh(q), sh(k), sh(v)
    att = (q @ k.transpose(0, 1, 3, 2)) / np.sqrt(d // heads)
    mask = jnp.tril(jnp.ones((T, T), bool))
    att = jax.nn.softmax(jnp.where(mask, att, -1e9), -1)
    return (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d) @ wo


def route(h, router, top_k):
    """Mixtral routing: top-k of router logits, softmax over the selected logits."""
    logits = h @ router                                       # (B,T,E)
    probs = jax.nn.softmax(logits, -1)
    top_val, top_idx = jax.lax.top_k(logits, top_k)
    gates = jax.nn.softmax(top_val, -1)
    E = router.shape[-1]
    dense = jnp.sum(jax.nn.one_hot(top_idx, E) * gates[..., None], axis=-2)  # (B,T,E)
    return logits, probs, top_idx, dense


def moe(h, lp, top_k):
    logits, probs, top_idx, dense = route(h, lp["router"], top_k)
    # dense evaluation of all experts, combined with the sparse gate (exact top-k math)
    a = jnp.einsum("btd,edf->btef", h, lp["w1"])
    b = jnp.einsum("btd,edf->btef", h, lp["w3"])
    y = jnp.einsum("btef,efd->bted", jax.nn.silu(a) * b, lp["w2"])
    out = jnp.einsum("bted,bte->btd", y, dense)
    E = logits.shape[-1]
    frac = jnp.mean(jnp.sum(jax.nn.one_hot(top_idx, E), -2), (0, 1)) / top_k
    aux = E * jnp.sum(frac * jnp.mean(probs, (0, 1)))
    return out, aux, logits, top_idx


def forward(params, ids, cfg=CFG, record=False):
    """Returns (logits, aux_loss[, record dict]).

    record=True additionally returns, per layer l, the hidden states needed for
    training-free *speculative* gating: x_in (layer input, before attention) and
    x_mid (after attention, i.e. the router input of layer l)."""
    T = ids.shape[1]
    x = params["emb"][ids] + params["pos"][:T]

    def layer(x, lp):
        x_in = x
        x = x + attention(rms(x, lp["ln1"]), lp["wqkv"], lp["wo"], cfg["heads"])
        x_mid = x
        y, aux, logits, top_idx = moe(rms(x, lp["ln2"]), lp, cfg["top_k"])
        x = x + y
        rec = (x_in, x_mid, logits, top_idx) if record else None
        return x, (aux, rec)

    x, (aux, rec) = jax.lax.scan(layer, x, params["layers"])
    logits = rms(x, params["lnf"]) @ params["emb"].T
    if record:
        return logits, aux.mean(), rec
    return logits, aux.mean()


def loss_fn(params, ids, cfg=CFG):
    logits, aux = forward(params, ids[:, :-1], cfg)
    tgt = ids[:, 1:]
    lp = jax.nn.log_softmax(logits, -1)
    ce = -jnp.mean(jnp.take_along_axis(lp, tgt[..., None], -1))
    return ce + cfg["aux_coef"] * aux, (ce, aux)


def save_params(params, path):
    flat = {"/".join(str(getattr(k, "key", k)) for k in kp): np.asarray(v)
            for kp, v in jax.tree_util.tree_flatten_with_path(params)[0]}
    np.savez(path, **flat)


def load_params(path):
    z = np.load(path)
    p = {"layers": {}}
    for k in z.files:
        parts = k.split("/")
        if parts[0] == "layers":
            p["layers"][parts[1]] = jnp.asarray(z[k])
        else:
            p[parts[0]] = jnp.asarray(z[k])
    return p


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
