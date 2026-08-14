"""Figures from timing.csv.

    python plot.py

Produces, from the median of the repeats:

  time_per_step.png   ms per MD step vs number of atoms, log-log, one line per engine,
                      SR dashed and LES solid
  les_overhead.png    LES cost as a percentage of the SR baseline, same engine, same size
  nmax_policy.png     the two N_max policies overlaid, for the engines that ran both
"""

import csv
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common as C  # noqa: E402

VARIANT_STYLE = {"sr": (":", "SR"), "les": ("-", "LES"), "les_u": ("--", "LES+dipole")}


def load():
    """median ms/step per (backbone, variant, engine, policy, natoms), ok rows only."""
    raw = defaultdict(list)
    with open(C.CSV_PATH, newline="") as fh:
        for r in csv.DictReader(fh):
            if r["status"] != "ok" or not r["s_per_step"]:
                continue
            key = (r["backbone"], r["variant"], r["engine"], r["nmax_policy"], int(r["natoms"]))
            raw[key].append(float(r["s_per_step"]) * 1e3)
    return {k: statistics.median(v) for k, v in raw.items()}


def curves(data, policy, backbone):
    """{(variant, engine): [(natoms, ms), ...]} sorted by size."""
    out = defaultdict(list)
    for (bb, variant, engine, pol, natoms), ms in data.items():
        if bb == backbone and pol == policy:
            out[(variant, engine)].append((natoms, ms))
    return {k: sorted(v) for k, v in out.items()}


def fig_time_per_step(data):
    backbones = sorted({k[0] for k in data})
    fig, axes = plt.subplots(1, len(backbones), figsize=(6 * len(backbones), 5), squeeze=False)
    for ax, backbone in zip(axes[0], backbones):
        cs = curves(data, "fixed", backbone)
        engines_seen = sorted({e for _, e in cs})
        colours = {e: f"C{i}" for i, e in enumerate(engines_seen)}
        for (variant, engine), pts in sorted(cs.items()):
            style, label = VARIANT_STYLE.get(variant, ("-", variant))
            n, ms = zip(*pts)
            ax.plot(n, ms, style, color=colours[engine], marker="o", ms=3,
                    label=f"{engine} · {label}")
        ax.set(xscale="log", yscale="log", xlabel="atoms", ylabel="ms per MD step",
               title=f"{backbone} — N_max as trained")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(C.HERE / "time_per_step.png", dpi=200)
    print("  time_per_step.png")


def fig_overhead(data):
    """LES / SR - 1, in percent: the apples-to-apples number, since the two share an engine,
    a structure and a neighbour-list cost."""
    fig, ax = plt.subplots(figsize=(7, 5))
    i = 0
    for backbone in sorted({k[0] for k in data}):
        cs = curves(data, "fixed", backbone)
        for variant in ("les", "les_u"):
            for engine in sorted({e for v, e in cs if v == variant}):
                sr = dict(cs.get(("sr", engine), []))
                les = dict(cs.get((variant, engine), []))
                shared = sorted(set(sr) & set(les))
                if not shared:
                    continue
                ax.plot(shared, [100 * (les[n] / sr[n] - 1) for n in shared],
                        marker="o", ms=3, color=f"C{i % 10}",
                        label=f"{backbone} · {engine} · {VARIANT_STYLE[variant][1]}")
                i += 1
    ax.axhline(0, color="k", lw=0.8)
    ax.set(xscale="log", xlabel="atoms", ylabel="LES overhead over SR [%]")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(C.HERE / "les_overhead.png", dpi=200)
    print("  les_overhead.png")


def fig_policy(data):
    """Both N_max policies, where both were run. `fixed` keeps a constant k-grid, so its cost
    is O(N); `scaled` keeps the accuracy the training cell had, and K grows with the cell, so
    the reciprocal sum becomes O(N^2)."""
    pairs = {(k[0], k[1], k[2]) for k in data if k[3] == "scaled"}
    if not pairs:
        print("  nmax_policy.png skipped: no scaled rows yet")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (backbone, variant, engine) in enumerate(sorted(pairs)):
        for policy, ls in (("fixed", "-"), ("scaled", "--")):
            pts = sorted((n, ms) for (bb, v, e, pol, n), ms in data.items()
                         if (bb, v, e, pol) == (backbone, variant, engine, policy))
            if pts:
                n, ms = zip(*pts)
                ax.plot(n, ms, ls, marker="o", ms=3, color=f"C{i % 10}",
                        label=f"{backbone} · {engine} · {variant} · N_max {policy}")
    ax.set(xscale="log", yscale="log", xlabel="atoms", ylabel="ms per MD step",
           title="k-grid held constant vs scaled to keep accuracy")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=5, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(C.HERE / "nmax_policy.png", dpi=200)
    print("  nmax_policy.png")


def main():
    if not C.CSV_PATH.exists():
        raise SystemExit(f"{C.CSV_PATH} not found -- run ./run_timing.sh first")
    data = load()
    if not data:
        raise SystemExit("no rows with status=ok in timing.csv")
    print(f"loaded {len(data)} (variant, engine, policy, size) points")
    fig_time_per_step(data)
    fig_overhead(data)
    fig_policy(data)


if __name__ == "__main__":
    main()
