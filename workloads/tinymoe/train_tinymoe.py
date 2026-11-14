"""Train TinyMoE on prose + code (CPU, JAX). ~45 min on a 2-core container (resumable).

Usage:  python3 train_tinymoe.py [--steps 1000]
Writes: checkpoint/tokenizer/, checkpoint/params.npz, checkpoint/train_log.json
"""
import argparse
import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import optax

import tinymoe as tm

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

os.makedirs(tm.CKPT, exist_ok=True)
tok = tm.load_tokenizer()
data = tm.domain_splits(tok)
ctx = tm.CFG["ctx"]
train = {d: data[d]["train"] for d in data}
val = {d: tm.chunk(data[d]["profile"], ctx, n=16) for d in data}
print({d: len(v) for d, v in train.items()}, "train tokens per domain")

rng = np.random.default_rng(args.seed)


def batch():
    """Half prose, half code windows sampled uniformly from the training split."""
    rows = []
    for i in range(args.batch):
        ids = train["prose" if i % 2 == 0 else "code"]
        s = rng.integers(0, len(ids) - ctx)
        rows.append(ids[s:s + ctx])
    return jnp.asarray(np.stack(rows))


params = tm.init_params(jax.random.PRNGKey(args.seed))
RESUME = os.path.join(tm.CKPT, "resume.pkl")
warm = min(100, max(1, args.steps // 10))
sched = optax.warmup_cosine_decay_schedule(0.0, args.lr, warm, args.steps, args.lr * 0.05)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.05))
state = opt.init(params)


@jax.jit
def step(params, state, ids):
    (loss, (ce, aux)), g = jax.value_and_grad(tm.loss_fn, has_aux=True)(params, ids)
    upd, state = opt.update(g, state, params)
    return optax.apply_updates(params, upd), state, ce, aux


@jax.jit
def eval_ce(params, ids):
    return tm.loss_fn(params, ids)[1][0]


log, start, t0 = [], 1, time.time()
if os.path.exists(RESUME):                       # resume after an interruption
    import pickle
    with open(RESUME, "rb") as f:
        ck = pickle.load(f)
    params, state, log, start = ck["params"], ck["state"], ck["log"], ck["step"] + 1
    rng.bit_generator.state = ck["rng"]
    t0 -= log[-1]["sec"] if log else 0
    print("resumed at step", start)
for it in range(start, args.steps + 1):
    params, state, ce, aux = step(params, state, batch())
    if it % 50 == 0 or it == 1:
        rec = {"step": it, "train_ce": float(ce), "aux": float(aux), "sec": round(time.time() - t0, 1)}
        if it % 250 == 0 or it == args.steps:
            rec.update({f"val_ce_{d}": float(eval_ce(params, jnp.asarray(v))) for d, v in val.items()})
        log.append(rec)
        print(rec, flush=True)
        tm.write_json({"cfg": tm.CFG, "args": vars(args), "log": log},
                      os.path.join(tm.CKPT, "train_log.json"))
    if it % 25 == 0:
        import pickle
        with open(RESUME + ".tmp", "wb") as f:
            pickle.dump({"params": jax.device_get(params), "state": jax.device_get(state),
                         "log": log, "step": it, "rng": rng.bit_generator.state}, f)
        os.replace(RESUME + ".tmp", RESUME)
tm.save_params(params, os.path.join(tm.CKPT, "params.npz"))
import shutil
shutil.copy(os.path.join(tm.CKPT, "train_log.json"), os.path.join(tm.HERE, "train_log.json"))  # committed copy
print("saved", os.path.join(tm.CKPT, "params.npz"))
