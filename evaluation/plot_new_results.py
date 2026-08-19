"""plot_new_results.py -- publication figures for the ballast-13 revision.

Generates (skipping any figure whose CSV is not yet present):
  figures/fig_q1_perf.pdf/.png        restyled Q1 capacity sweep w/ 95% bands
  figures/fig_margin_attribution.pdf  2026 margin decomposition (E2)
  figures/fig_security_sensitivity.pdf delta/eta sensitivity + tightness (E3)
  figures/fig_deployable.pdf          deployable-vs-deployable (E6)

Style: Okabe-Ito series colors matched to the paper's TikZ palette, thin
marks, no top/right spines, y-grid only, direct labels for the headline
series plus a frameless legend.
"""

import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))

C = {"Ballast": "#0072B2", "Shaduf": "#E69F00", "Horcrux": "#009E73",
     "Revive": "#CC79A7", "LN": "#333333", "Ballast-PC": "#7F7F7F"}
MK = {"Ballast": "o", "Shaduf": "s", "Horcrux": "^", "Revive": "d",
      "LN": "x", "Ballast-PC": "v"}
LS = {"Ballast": "-", "Shaduf": "-", "Horcrux": "-", "Revive": "-",
      "LN": "-", "Ballast-PC": "-"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 9,
    "legend.fontsize": 7.2, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def _read(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return list(csv.DictReader(f))


def _grid(ax):
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    ax.set_axisbelow(True)


def fig_q1():
    rows = _read("q1_sweep_seeds.csv")
    if not rows:
        print("q1_sweep_seeds.csv not ready; skip fig_q1")
        return
    data = defaultdict(lambda: defaultdict(list))  # scheme -> cap -> [ratios]
    for r in rows:
        data[r["scheme"]][int(r["capacity"])].append(float(r["success_ratio"]))
    order = ["Ballast", "Horcrux", "Shaduf", "Ballast-PC", "Revive", "LN"]
    fig, ax = plt.subplots(figsize=(4.4, 2.3))
    for name in order:
        caps = sorted(data[name])
        m = np.array([np.mean(data[name][c]) for c in caps]) * 100
        sd = np.array([np.std(data[name][c], ddof=1) for c in caps]) * 100
        ci = 1.96 * sd / np.sqrt([len(data[name][c]) for c in caps])
        ax.plot(caps, m, color=C[name], marker=MK[name], ls=LS[name],
                lw=1.8, ms=3.6, mew=0.8, mec="white",
                label=name, zorder=5 if name == "Ballast" else 3)
        ax.fill_between(caps, m - ci, m + ci, color=C[name], alpha=0.14, lw=0)
    # direct labels for the headline pair at the right edge
    for name, dy in [("Ballast", 5), ("Horcrux", -6)]:
        caps = sorted(data[name])
        yv = np.mean(data[name][caps[-1]]) * 100
        ax.annotate(name, (caps[-1], yv), xytext=(4, dy),
                    textcoords="offset points", fontsize=7.2,
                    color=C[name], va="center")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(r["capacity"]) for r in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("channel capacity scale")
    ax.set_ylabel("payment success (\\%)" if plt.rcParams.get("text.usetex")
                  else "payment success (%)")
    _grid(ax)
    ax.legend(frameon=False, ncol=3, loc="lower right",
              columnspacing=1.0, handlelength=1.7)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGURES, f"fig_q1_perf.{ext}"))
    print("wrote fig_q1_perf")


def fig_margin():
    rows = _read("margin_attribution.csv")
    if not rows:
        print("margin_attribution.csv not ready; skip")
        return
    margins, nodes, cis = {}, {}, {}
    for r in rows:
        v = r["variant"]
        margins[v] = float(r["margin_vs_horcrux_pp"])
        nodes[v] = int(r["nodes"])
        if r["scheme"] in ("Ballast", "Horcrux"):
            cis.setdefault(v, 0.0)
            cis[v] += (float(r["ci95"]) * 100) ** 2
    order = ["G2021", "H_scale", "H_topology", "H_capacity", "G2026"]
    labels = {
        "G2021": "2021 snapshot",
        "G2026": "2026 snapshot",
        "H_capacity": "2026 topology\n+ 2021 capacities",
        "H_topology": "2021 topology\n+ 2026 capacities",
        "H_scale": "2021 subgraph\nat 2026 scale",
    }
    fig, ax = plt.subplots(figsize=(4.4, 2.1))
    ys = np.arange(len(order))[::-1]
    vals = [margins[v] for v in order]
    colors = ["#0072B2" if v.startswith("G") else "#7FB3D5" for v in order]
    errs = [cis.get(v, 0.0) ** 0.5 for v in order]
    ax.barh(ys, vals, height=0.58, color=colors, edgecolor="white", lw=0.5,
            xerr=errs, error_kw=dict(ecolor="black", lw=0.8, capsize=2))
    ax.set_xlim(0, max(vals) * 1.18)
    for y, v, e in zip(ys, vals, errs):
        ax.annotate(f"{v:.2f}", (v + e, y), xytext=(9, 0),
                    textcoords="offset points", va="center", fontsize=7.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[v] for v in order], fontsize=7.2)
    ax.set_xlabel("Ballast $-$ Horcrux margin (pp)")
    ax.axvline(0, color="black", lw=0.7)
    _grid(ax)
    ax.grid(False, axis="y")
    fig.savefig(os.path.join(FIGURES, "fig_margin_attribution.pdf"))
    fig.savefig(os.path.join(FIGURES, "fig_margin_attribution.png"))
    print("wrote fig_margin_attribution")


def fig_security():
    rows = _read("security_delta_eta.csv")
    if not rows:
        print("security_delta_eta.csv not ready; skip")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.1))

    # (a) tightness ratio vs tau_e per delta, at the paper eta instance (40/s)
    ax = axes[0]
    base = [r for r in rows if r["counterparties"] == "8"
            and r["inflight"] == "1" and r["r_min_s"] == "0.2"]
    deltas = sorted({float(r["delta_s"]) for r in base})
    blues = plt.cm.Blues(np.linspace(0.45, 0.95, len(deltas)))
    for d, col in zip(deltas, blues):
        pts = sorted((float(r["tau_e_s"]), float(r["ratio"]))
                     for r in base if float(r["delta_s"]) == d)
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col,
                marker="o", ms=3.2, lw=1.6, mec="white", mew=0.6,
                label=f"$\\delta$={d:g}s")
    ax.axhline(1.0, color="black", lw=0.7, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("checkpoint epoch $\\tau_e$ (s)")
    ax.set_ylabel("attack / bound ratio")
    ax.set_ylim(0.3, 1.08)
    ax.legend(frameon=False, loc="lower right", ncol=2,
              columnspacing=0.9, handlelength=1.4)
    _grid(ax)

    # (b) linear scaling in eta at tau_e=10s, delta=0.2
    ax = axes[1]
    sel = [r for r in rows if r["tau_e_s"] == "10.0" and r["delta_s"] == "0.2"]
    etas = sorted({int(r["eta_per_s"]) for r in sel})
    att = [next(float(r["attack_sat"]) for r in sel
                if int(r["eta_per_s"]) == e) / 1e9 for e in etas]
    bnd = [next(float(r["bound_sat"]) for r in sel
                if int(r["eta_per_s"]) == e) / 1e9 for e in etas]
    ax.plot(etas, bnd, color="#7F7F7F", lw=1.4, ls="--", marker="s", ms=3.0,
            mec="white", mew=0.6, label="bound $\\theta\\eta(\\tau_e{+}\\delta)$")
    ax.plot(etas, att, color="#0072B2", lw=1.8, marker="o", ms=3.4,
            mec="white", mew=0.6, label="greedy attack")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("admission rate $\\eta$ (draws/s)")
    ax.set_ylabel("exposure (B sat)")
    ax.legend(frameon=False, loc="upper left")
    _grid(ax)

    fig.tight_layout(w_pad=2.0)
    fig.savefig(os.path.join(FIGURES, "fig_security_sensitivity.pdf"))
    fig.savefig(os.path.join(FIGURES, "fig_security_sensitivity.png"))
    print("wrote fig_security_sensitivity")


def fig_deployable():
    rows = _read("deployable_compare.csv")
    if not rows:
        print("deployable_compare.csv not ready; skip")
        return
    # judge readiness: paper run has ci95 > 0 rows
    p = "0.99"
    sel = [r for r in rows if r["p"] == p]
    if not sel:
        return
    series = {
        "ballast_adaptive": ("Ballast", "Ballast"),
        "horcrux": ("Horcrux", "Horcrux"),
        "shaduf": ("Shaduf", "Shaduf"),
        "revive": ("Revive", "Revive"),
    }
    fig, ax = plt.subplots(figsize=(4.4, 1.7))
    ends = []
    for key, (label, cname) in series.items():
        pts = sorted((float(r["L_rtt"]), float(r["effective_success"]) * 100,
                      float(r["ci95"]) * 100)
                     for r in sel if r["variant"] == key)
        x = [q[0] for q in pts]; y = [q[1] for q in pts]; ci = [q[2] for q in pts]
        ax.plot(x, y, color=C[cname], marker=MK[cname], ls=LS[cname],
                lw=1.8, ms=3.6, mec="white", mew=0.8,
                zorder=5 if cname == "Ballast" else 3)
        ax.fill_between(x, np.array(y) - ci, np.array(y) + ci,
                        color=C[cname], alpha=0.14, lw=0)
        ends.append((x[-1], y[-1], label, cname))
    # direct labels at the right end of each curve instead of a legend
    for x1, y1, label, cname in ends:
        ax.annotate(label, (x1, y1), xytext=(5, 0),
                    textcoords="offset points", va="center",
                    fontsize=7.5, color=C[cname], fontweight="bold")
    ax.set_xlim(right=max(e[0] for e in ends) * 1.22)
    ax.set_xlabel("coin-shift latency $L$ (RTT units), liveness $p=0.99$")
    ax.set_ylabel("effective success (%)")
    _grid(ax)
    fig.savefig(os.path.join(FIGURES, "fig_deployable.pdf"))
    fig.savefig(os.path.join(FIGURES, "fig_deployable.png"))
    print("wrote fig_deployable")


def fig_phi():
    rows = _read("phi_sweep.csv")
    if not rows:
        print("phi_sweep.csv not ready; skip")
        return
    fig, ax = plt.subplots(figsize=(4.4, 2.15))
    for key, label in (("Ballast", "Ballast"), ("Ballast-perch", "Ballast-PC")):
        pts = sorted((float(r["phi"]), float(r["success_ratio"]) * 100)
                     for r in rows if r["scheme"] == key and float(r["phi"]) > 0)
        cname = "Ballast" if key == "Ballast" else "Ballast-PC"
        ax.plot([q[0] for q in pts], [q[1] for q in pts], color=C[cname],
                marker=MK[cname], ls=LS[cname], lw=1.8, ms=3.6,
                mec="white", mew=0.8, label=label,
                zorder=5 if cname == "Ballast" else 3)
    for key in ("Horcrux", "Shaduf", "LN"):
        r0 = next(r for r in rows if r["scheme"] == key)
        ax.axhline(float(r0["success_ratio"]) * 100, color=C[key],
                   ls=LS[key], lw=1.2, label=key)
    ax.axvline(0.30, color="black", lw=0.6, ls=":")
    ax.annotate("$\\phi=0.30$", (0.30, 44), xytext=(4, 0),
                textcoords="offset points", fontsize=6.8, color="black")
    ax.set_xlabel("skim fraction $\\phi$")
    ax.set_ylabel("payment success (%)")
    _grid(ax)
    ax.legend(frameon=False, ncol=2, loc="center right", handlelength=1.7)
    fig.savefig(os.path.join(FIGURES, "fig_phi_sweep.pdf"))
    fig.savefig(os.path.join(FIGURES, "fig_phi_sweep.png"))
    print("wrote fig_phi_sweep")


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    fig_q1()
    fig_margin()
    fig_security()
    fig_deployable()
    fig_phi()


def fig_margin_phi():
    """Combined (a) margin attribution + (b) phi sweep, stacked vertically."""
    mrows = _read("margin_attribution.csv")
    prows = _read("phi_sweep.csv")
    if not mrows or not prows:
        print("inputs missing; skip fig_margin_phi")
        return
    fig, (axa, axb) = plt.subplots(2, 1, figsize=(4.4, 3.6),
                                   gridspec_kw={"height_ratios": [1.0, 1.05],
                                                "hspace": 0.52})
    # (a) margin attribution
    margins, cis = {}, {}
    for r in mrows:
        v = r["variant"]
        margins[v] = float(r["margin_vs_horcrux_pp"])
        if r["scheme"] in ("Ballast", "Horcrux"):
            cis[v] = cis.get(v, 0.0) + (float(r["ci95"]) * 100) ** 2
    order = ["G2021", "H_scale", "H_topology", "H_capacity", "G2026"]
    labels = {"G2021": "2021 snapshot", "G2026": "2026 snapshot",
              "H_capacity": "2026 topo + 2021 cap.",
              "H_topology": "2021 topo + 2026 cap.",
              "H_scale": "2021 sub @ 2026 scale"}
    ys = np.arange(len(order))[::-1]
    vals = [margins[v] for v in order]
    errs = [cis.get(v, 0.0) ** 0.5 for v in order]
    colors = ["#0072B2" if v.startswith("G") else "#7FB3D5" for v in order]
    axa.barh(ys, vals, height=0.6, color=colors, edgecolor="white", lw=0.5,
             xerr=errs, error_kw=dict(ecolor="black", lw=0.8, capsize=2))
    axa.set_xlim(0, max(vals) * 1.2)
    for y, v, e in zip(ys, vals, errs):
        axa.annotate(f"{v:.2f}", (v + e, y), xytext=(8, 0),
                     textcoords="offset points", va="center", fontsize=7)
    axa.set_yticks(ys)
    axa.set_yticklabels([labels[v] for v in order], fontsize=6.8)
    axa.set_xlabel("(a) Ballast $-$ Horcrux margin (pp)", fontsize=7.8)
    _grid(axa); axa.grid(False, axis="y")
    # (b) phi sweep
    pts = {}
    base = {}
    for r in prows:
        phi = float(r["phi"]); sr = float(r["success_ratio"]) * 100
        if phi < 0:
            base[r["scheme"]] = sr
        else:
            pts.setdefault(r["scheme"], []).append((phi, sr))
    if "Ballast-perch" in pts:
        pts["Ballast-PC"] = pts.pop("Ballast-perch")
    for name in ("Ballast", "Ballast-PC"):
        if not pts.get(name):
            continue
        p = sorted(pts[name])
        axb.plot([q[0] for q in p], [q[1] for q in p], color=C[name],
                 marker=MK[name], ls=LS[name], lw=1.8, ms=3.4,
                 mec="white", mew=0.7, label=name)
    for name, dy in (("Horcrux", 1.2), ("Shaduf", -1.2), ("LN", 1.2)):
        if name in base:
            axb.axhline(base[name], color=C[name], ls=LS[name], lw=1.2)
            axb.annotate(name, (0.335 if name != "LN" else 0.115, base[name] + dy), fontsize=6.8,
                         color=C[name], va="bottom" if dy > 0 else "top")
    axb.axvline(0.30, color="black", ls=":", lw=0.8)
    axb.annotate("default $\\phi$", (0.305, 48), fontsize=6.5, color="black")
    axb.set_xlim(0.08, 0.53)
    axb.set_xlabel("(b) skim fraction $\\phi$", fontsize=7.8)
    axb.set_ylabel("success (%)")
    axb.legend(frameon=False, fontsize=6.8, loc="center right")
    _grid(axb)
    fig.savefig(os.path.join(FIGURES, "fig_margin_phi.pdf"))
    fig.savefig(os.path.join(FIGURES, "fig_margin_phi.png"))
    print("wrote fig_margin_phi")
