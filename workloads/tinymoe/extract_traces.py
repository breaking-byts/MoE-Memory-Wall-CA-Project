"""Record TinyMoE routing on held-out text -> workloads/traces/tinymoe_{prose,code}.npz

For every held-out token and layer we store
  ids     the top-2 experts the router actually chose (rank order)
  spec0   softmax(router_l(norm_l(x_in_l)))       router of layer l on the layer's input,
                                                  i.e. before its attention (APEX-style timing)
  spec1   softmax(router_l(norm_l(x_mid_{l-1})))  router of layer l on the hidden state that
                                                  layer l-1's router saw (speculative gating,
                                                  Eliseev & Mazur 2023): known 1 layer ahead
  spec2   same with x_mid_{l-2}: known 2 layers ahead
  calib{d} histogram calibration P(routed | speculative rank, softmax bin), counted on the
           profiling split (prefetch.predictors.spec_calibration)
All three re-use the model's own router weights -- no training, no extra parameters.

The forward pass over a held-out window is teacher-forced; with causal attention the
routing of position t depends only on tokens <= t, so it is exactly the routing a
decoder would produce while generating that text token by token.
"""
import os
import numpy as np
import jax
import jax.numpy as jnp

import tinymoe as tm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(tm.HERE)))
from prefetch import predictors as P  # noqa: E402

N_SEQ = 32           # windows per split and domain
PROMPT = 64          # first 64 tokens = prefill history, remaining 192 = decode steps
OUT = os.path.join(os.path.dirname(tm.HERE), "traces")


@jax.jit
def record(params, ids):
    _, _, (x_in, x_mid, logits, top_idx) = tm.forward(params, ids, record=True)
    lp = params["layers"]
    norm = lambda x, g: tm.rms(x, g[:, None, None, :])
    route = lambda h: jax.nn.softmax(jnp.einsum("lbtd,lde->lbte", h, lp["router"]), -1)
    spec0 = route(norm(x_in, lp["ln2"]))
    shift = lambda x, d: jnp.concatenate([jnp.zeros_like(x[:d]), x[:-d]], 0)   # x[l-d] at index l
    spec1 = route(norm(shift(x_mid, 1), lp["ln2"]))
    spec2 = route(norm(shift(x_mid, 2), lp["ln2"]))
    return top_idx, spec0, spec1, spec2


def run(params, windows, bs=4):
    outs = [record(params, jnp.asarray(windows[i:i + bs])) for i in range(0, len(windows), bs)]
    cat = lambda j: np.concatenate([np.asarray(o[j]) for o in outs], axis=1)   # L,S,T,*
    ids, s0, s1, s2 = (cat(j).transpose(1, 2, 0, 3) for j in range(4))        # S,T,L,*
    return ids, s0, s1, s2


def main():
    params = tm.load_params(os.path.join(tm.CKPT, "params.npz"))
    tok = tm.load_tokenizer()
    data = tm.domain_splits(tok)
    os.makedirs(OUT, exist_ok=True)
    q = lambda p: np.round(np.asarray(p) * 255).astype(np.uint8)
    for dom in ("prose", "code"):
        ev = tm.chunk(data[dom]["eval"], tm.CFG["ctx"], n=N_SEQ)
        pr = tm.chunk(data[dom]["profile"], tm.CFG["ctx"], n=N_SEQ)
        ids, s0, s1, s2 = run(params, ev)
        pids, ps0, ps1, ps2 = run(params, pr)
        pids = pids.astype(np.int64)
        deq = lambda x: (np.round(np.asarray(x) * 255) / 255).astype(np.float32)   # as stored
        # profiling split only: calibration of the speculative router outputs
        calib = {d: P.spec_calibration(pids, deq(x), d) for d, x in ((0, ps0), (1, ps1), (2, ps2))}
        s1[:, :, :1] = 0
        s2[:, :, :2] = 0                                # undefined for the first d layers
        path = os.path.join(OUT, f"tinymoe_{dom}.npz")
        np.savez_compressed(path, ids=ids.astype(np.int8), profile_ids=pids.astype(np.int8),
                            spec0=q(s0), spec1=q(s1), spec2=q(s2), prompt_len=PROMPT,
                            calib0=calib[0], calib1=calib[1], calib2=calib[2],
                            tokens=ev.astype(np.int16))
        print(dom, ids.shape, "->", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
