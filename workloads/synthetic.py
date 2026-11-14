"""Synthetic routing workloads.

uniform              LLMSimulator's default routing (each token draws top-2 of 8 uniformly):
                     the null hypothesis, no locality at all.

mixtral-calibrated   Latent-logit model whose *temporal* locality is calibrated to the
                     published Mixtral-8x7B measurement (Jiang et al. 2024, Table 5:
                     share of consecutive tokens that keep at least one of their two
                     experts, averaged over the 8 Pile domains: 48.0 % at layer 0,
                     63.7 % at layer 15, 50.9 % at layer 31; random = 46.4 %).
                     Router logits of token t at layer l:
                         z[t,l] = a_l * z[t-1,l] + beta * P_l z[t,l-1] + noise
                     a_l (per layer) is fitted so the generated trace reproduces the
                     Table-5 curve (linear interpolation between layers 0/15/31).
                     beta (cross-layer "residual stream" strength, P_l a fixed random
                     permutation per layer) is NOT published for Mixtral.  Because the
                     cross-layer term also creates token-to-token repetition, beta > 0.7
                     makes it impossible to match Table 5 (layer-15 repetition overshoots
                     0.637 even with a_l = 0).  We therefore use beta = 0.7: the strongest
                     cross-layer coupling still consistent with Mixtral's published
                     temporal statistics.  experiments/f1_predictor_accuracy.py sweeps it.
                     spec{1,2} = softmax of the conditional mean of z[t,l] given the
                     state d layers earlier = what a router applied to an earlier hidden
                     state can know.

Usage: python3 synthetic.py [--beta 0.7]
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prefetch import predictors as P  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
E, K, L = 8, 2, 32
S, T, PROMPT = 32, 256, 64
TABLE5_FIRST_OR_SECOND = {0: 0.4804, 15: 0.6369, 31: 0.5091}
TABLE5_FIRST = {0: 0.1409, 15: 0.2640, 31: 0.2268}


def target_curve(tab):
    xs = sorted(tab)
    return np.interp(np.arange(L), xs, [tab[x] for x in xs])


def generate(a, beta, n_seq, rng, perms, bias):
    z = np.zeros((n_seq, T, L, E))
    hat1 = np.zeros_like(z)
    hat2 = np.zeros_like(z)
    for t in range(T):
        for l in range(L):
            m = np.zeros((n_seq, E))
            if t > 0:
                m += a[l] * z[:, t - 1, l]
            if l > 0:
                m += beta * z[:, t, l - 1][:, perms[l]]
            hat1[:, t, l] = m + bias[l]
            if l > 1:                                   # mean given layer l-2 (and t-1)
                prev = (a[l - 1] * z[:, t - 1, l - 1] if t > 0 else 0) + beta * z[:, t, l - 2][:, perms[l - 1]]
                m2 = (a[l] * z[:, t - 1, l] if t > 0 else 0) + beta * prev[:, perms[l]]
                hat2[:, t, l] = m2 + bias[l]
            noise_sd = np.sqrt(max(1e-3, 1 - a[l] ** 2 - beta ** 2))
            z[:, t, l] = m + noise_sd * rng.standard_normal((n_seq, E))
            z[:, t, l] /= z[:, t, l].std(-1, keepdims=True) + 1e-9
    logits = z + bias[None, None]
    ids = np.argsort(-logits, -1)[..., :K]
    return ids, hat1, hat2


def softmax(x, temp):
    x = x / temp
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def rep_first_or_second(ids):
    oh = np.zeros(ids.shape[:-1] + (E,))
    np.put_along_axis(oh, ids, 1, -1)
    return ((oh[:, 1:] * oh[:, :-1]).sum(-1) >= 1).mean((0, 1))


def build_mixtral_calibrated(beta, seed=0, iters=12):
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(E) for _ in range(L)]
    bias = 0.15 * rng.standard_normal((L, E))           # mild popularity skew
    target = target_curve(TABLE5_FIRST_OR_SECOND)
    a = np.full(L, 0.3)
    for it in range(iters):                              # fit a_l to Table 5
        ids, _, _ = generate(a, beta, 24, np.random.default_rng(100 + it), perms, bias)
        got = rep_first_or_second(ids)
        a = np.clip(a + 1.5 * (target - got), 0.0, np.sqrt(max(0.0, 0.98 - beta ** 2)))
    ids, h1, h2 = generate(a, beta, 2 * S, np.random.default_rng(seed + 7), perms, bias)
    got = rep_first_or_second(ids[:S])
    first = (ids[:S, 1:, :, 0] == ids[:S, :-1, :, 0]).mean((0, 1))
    temp = np.sqrt(max(1e-3, 1 - beta ** 2))             # noise scale of the logits
    q = lambda p: np.round(p * 255).astype(np.uint8)
    s1 = softmax(h1, temp)
    s1[:, :, :1] = 0
    s2 = softmax(h2, temp)
    s2[:, :, :2] = 0
    deq = lambda x: (q(x) / 255).astype(np.float32)          # as stored
    prof = ids[S:].astype(np.int64)
    calib = {1: P.spec_calibration(prof, deq(s1[S:]), 1), 2: P.spec_calibration(prof, deq(s2[S:]), 2)}
    out = dict(ids=ids[:S].astype(np.int8), profile_ids=ids[S:].astype(np.int8),
               spec1=q(s1[:S]), spec2=q(s2[:S]), calib1=calib[1], calib2=calib[2],
               prompt_len=PROMPT, a=a, beta=beta,
               fit_first_or_second=got, target_first_or_second=target,
               fit_first=first, target_first=target_curve(TABLE5_FIRST))
    return out


def build_uniform(seed=0):
    rng = np.random.default_rng(seed)
    ids = np.argsort(rng.random((2 * S, T, L, E)), -1)[..., :K]
    return dict(ids=ids[:S].astype(np.int8), profile_ids=ids[S:].astype(np.int8), prompt_len=PROMPT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--beta", type=float, default=0.7)
    args = ap.parse_args()
    os.makedirs(os.path.join(HERE, "traces"), exist_ok=True)
    np.savez_compressed(os.path.join(HERE, "traces", "uniform.npz"), **build_uniform())
    mc = build_mixtral_calibrated(args.beta)
    np.savez_compressed(os.path.join(HERE, "traces", "mixtral_calibrated.npz"), **mc)
    for l in (0, 15, 31):
        print(f"layer {l:2d}: first-or-second repetition {mc['fit_first_or_second'][l]:.3f} "
              f"(Mixtral {mc['target_first_or_second'][l]:.3f}), first choice "
              f"{mc['fit_first'][l]:.3f} (Mixtral {mc['target_first'][l]:.3f})")
