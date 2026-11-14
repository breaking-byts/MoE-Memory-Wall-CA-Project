"""F0 (extends midterm D4): temporal and spatial correlation of expert accesses on the
final workloads, compared with the published Mixtral-8x7B statistics.

-> results/f0_correlation.csv, figures/f0_correlation.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: F401,F403
from prefetch.evaluate import correlation_stats  # noqa: E402

# Jiang et al. 2024 (Mixtral of Experts), Table 5, mean of the 8 Pile domains
MIXTRAL_T5 = {"first": {0: 0.1409, 15: 0.2640, 31: 0.2268},
              "first_or_second": {0: 0.4804, 15: 0.6369, 31: 0.5091}}
RANDOM_T5 = {"first": 1 / 8, "first_or_second": 1 - (6 / 8) * (5 / 7)}


def nmi(joint):
    p = joint / joint.sum()
    px, py = p.sum(1, keepdims=True), p.sum(0, keepdims=True)
    nz = p > 0
    mi = (p[nz] * np.log(p[nz] / (px @ py)[nz])).sum()
    h = -(px[px > 0] * np.log(px[px > 0])).sum()
    return mi / h


def top1_joint_layers(ids, l):
    j = np.zeros((8, 8))
    np.add.at(j, (ids[:, :, l, 0].ravel(), ids[:, :, l + 1, 0].ravel()), 1)
    return j


def top1_joint_tokens(ids, l):
    j = np.zeros((8, 8))
    np.add.at(j, (ids[:, :-1, l, 0].ravel(), ids[:, 1:, l, 0].ravel()), 1)
    return j


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    stats = {}
    for w in WORKLOADS:
        wl = Workload(w)
        st = correlation_stats(wl.ids)
        stats[w] = st
        nmi_l = np.mean([nmi(top1_joint_layers(wl.ids, l)) for l in range(31)])
        nmi_t = np.mean([nmi(top1_joint_tokens(wl.ids, l)) for l in range(32)])
        # coverage: distinct experts per layer for a batch of 4 / 8 sequences (same step)
        cov = {B: np.mean([len(np.unique(wl.ids[g * B:(g + 1) * B, t, l]))
                           for g in range(wl.S // B) for t in range(64, 256, 8) for l in range(32)])
               for B in (1, 4, 8)}
        row = {"workload": w, "top_expert_share": float((st["freq"].max(1) / wl.K).mean()),
               "token_overlap": st["token_overlap"], "layer_overlap_same_id": st["layer_overlap_same_id"],
               "nmi_top1_layer_l_to_l+1": float(nmi_l), "nmi_top1_token_t_to_t+1": float(nmi_t),
               "distinct_experts_B1": float(cov[1]), "distinct_experts_B4": float(cov[4]),
               "distinct_experts_B8": float(cov[8])}
        for l in (0, 15, 31):
            row[f"rep_first_L{l}"] = float(st["rep_first"][l])
            row[f"rep_first_or_second_L{l}"] = float(st["rep_first_or_second"][l])
        rows.append(row)
    ref = {"workload": "Mixtral-8x7B (published, Table 5)"}
    for l in (0, 15, 31):
        ref[f"rep_first_L{l}"] = MIXTRAL_T5["first"][l]
        ref[f"rep_first_or_second_L{l}"] = MIXTRAL_T5["first_or_second"][l]
    rows.append(ref)
    rows.append({"workload": "random routing (null)", "token_overlap": 0.25, "layer_overlap_same_id": 0.25,
                 "nmi_top1_layer_l_to_l+1": 0.0, "nmi_top1_token_t_to_t+1": 0.0,
                 **{f"rep_first_L{l}": RANDOM_T5["first"] for l in (0, 15, 31)},
                 **{f"rep_first_or_second_L{l}": RANDOM_T5["first_or_second"] for l in (0, 15, 31)}})
    write_csv(rows, os.path.join(RESULTS, "f0_correlation.csv"))

    fig, ax = plt.subplots(len(WORKLOADS), 4, figsize=(18, 4 * len(WORKLOADS)), constrained_layout=True)
    for i, w in enumerate(WORKLOADS):
        st, wl = stats[w], Workload(w)
        im = ax[i, 0].imshow(st["freq"], aspect="auto", cmap="viridis", vmin=0, vmax=0.5)
        ax[i, 0].set_title(f"{w}\nexpert frequency per layer")
        ax[i, 0].set_xlabel("expert"); ax[i, 0].set_ylabel("layer")
        fig.colorbar(im, ax=ax[i, 0])
        j = top1_joint_layers(wl.ids, 15)
        im = ax[i, 1].imshow(j / j.sum(1, keepdims=True), vmin=0, vmax=0.8, cmap="magma")
        ax[i, 1].set_title("spatial: P(top-1 at layer 16 | top-1 at 15)")
        ax[i, 1].set_xlabel("expert at l+1"); ax[i, 1].set_ylabel("expert at l")
        fig.colorbar(im, ax=ax[i, 1])
        j = top1_joint_tokens(wl.ids, 15)
        im = ax[i, 2].imshow(j / j.sum(1, keepdims=True), vmin=0, vmax=0.8, cmap="magma")
        ax[i, 2].set_title("temporal: P(top-1 at t+1 | top-1 at t), layer 15")
        ax[i, 2].set_xlabel("expert at t+1"); ax[i, 2].set_ylabel("expert at t")
        fig.colorbar(im, ax=ax[i, 2])
        a = ax[i, 3]
        a.plot(st["rep_first_or_second"], label="first-or-second (this trace)")
        a.plot(st["rep_first"], label="first choice (this trace)")
        for key, mk in (("first_or_second", "s"), ("first", "o")):
            xs = sorted(MIXTRAL_T5[key])
            a.plot(xs, [MIXTRAL_T5[key][x] for x in xs], mk, c="k", label=f"Mixtral Table 5 ({key})")
            a.axhline(RANDOM_T5[key], ls=":", c="gray")
        a.set_ylim(0, 1); a.set_xlabel("layer"); a.set_title("consecutive-token expert repetition")
        a.legend(fontsize=7)
    fig.savefig(os.path.join(FIGURES, "f0_correlation.png"), dpi=100)
    print("wrote experiments/figures/f0_correlation.png")
    for r in rows:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
