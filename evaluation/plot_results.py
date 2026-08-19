"""
plot_results.py -- turn results/*.csv into the paper's figures and table bodies.

Outputs
  figures/fig_latency.pdf   -> Q2 sensitivity companion figure
  figures/fig_q1_perf.pdf   -> Q1 matched-capital ablation
  figures/fig_robustness.pdf -> phi sensitivity + temporal snapshots
  prints LaTeX row bodies for Table tab:perf (Q1) and Table tab:theta (Q3)

Run run_experiments.py first.
"""

import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGURES = os.path.join(HERE, "..", "figures")

# Okabe-Ito palette, matched to the paper's TikZ figures (ballast-main.tex):
#   Ballast=okBlue #0072B2, Shaduf=okOrange #E69F00, Horcrux=okGreen #009E73,
#   Ballast-PC/LN=okGray #7F7F7F / black.  Line style + marker also encode each
#   series so the figures stay legible in grayscale.
STYLE = {
    "Ballast": dict(color="#0072B2", marker="o", lw=2.2, zorder=5),
    "Shaduf":  dict(color="#E69F00", marker="s", lw=1.6, ls="--"),
    "Horcrux": dict(color="#009E73", marker="^", lw=1.6, ls="-."),
    "LN":      dict(color="#333333", marker="x", lw=1.4, ls=":"),
    "Revive":  dict(color="#CC79A7", marker="d", lw=1.4, ls="--"),
    "Ballast-PC": dict(color="#7F7F7F", marker="s", lw=1.7, ls="--"),
}


def _read(name):
    with open(os.path.join(RESULTS, name)) as f:
        return list(csv.DictReader(f))


def plot_latency():
    rows = _read("q2_sensitivity.csv")
    schemes = {}
    for r in rows:
        schemes.setdefault(r["scheme"], []).append(
            (float(r["L"]), float(r["S_central"])))
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for name, pts in schemes.items():
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, label=name, **STYLE.get(name, {}))
    ax.set_xlabel("coin-shift latency $L$ (round trips; on-chain $\\approx 600$)")
    ax.set_ylabel("effective success ratio")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_latency.pdf")
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def plot_q1():
    rows = _read("e1_pooling.csv")
    schemes = {}
    for r in rows:
        name = "Ballast-PC" if r["scheme"] == "Ballast-perch" else r["scheme"]
        schemes.setdefault(name, []).append(
            (int(r["capacity"]), float(r["success_ratio"])))
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    for name, pts in schemes.items():
        pts.sort()
        ax.plot([p[0] for p in pts], [p[1] for p in pts],
                label=name, **STYLE.get(name, {}))
    ax.set_xlabel("channel capacity factor")
    ax.set_ylabel("success ratio ($L=0$)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIGURES, "fig_q1_perf.pdf")
    os.makedirs(FIGURES, exist_ok=True)
    fig.savefig(out)
    print(f"wrote {out}")


def plot_robustness():
    """Two-panel robustness figure requested by the artifact review.

    Left: matched-capital skim-fraction sensitivity.  Right: the same schemes
    on independently reconstructed historical and current LN snapshots.
    """
    phi_rows = _read("phi_sweep.csv")
    snap_rows = _read("snapshot_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ax = axes[0]
    for name in ["Ballast", "Ballast-perch"]:
        pts = sorted((float(r["phi"]), float(r["success_ratio"]))
                     for r in phi_rows if r["scheme"] == name and float(r["phi"]) >= 0)
        label = "Ballast-PC" if name == "Ballast-perch" else name
        style = STYLE["Ballast"] if name == "Ballast" else STYLE["Ballast-PC"]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], label=label, **style)
    for name in ["LN", "Shaduf", "Horcrux"]:
        row = next(r for r in phi_rows if r["scheme"] == name)
        ax.axhline(float(row["success_ratio"]), label=name,
                   color=STYLE[name]["color"], ls=STYLE[name].get("ls", ":"), lw=1.0)
    ax.axvline(0.30, color="#777777", lw=0.8, ls=":")
    ax.set_xlabel("bond skim fraction $\\phi$")
    ax.set_ylabel("payment success ratio")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    available = {r["snapshot"] for r in snap_rows}
    preferred = [
        ("ln_600000", "600k"),
        ("ln_640000", "640k"),
        ("ln_677167", "2021\n677k"),
        ("ln_20230716", "2023-07"),
        ("ln_20260804_rgs", "2026-08\nRGS"),
    ]
    selected = [(snapshot, label) for snapshot, label in preferred
                if snapshot in available]
    known = {snapshot for snapshot, _label in selected}
    selected += [(snapshot, snapshot.removeprefix("ln_"))
                 for snapshot in sorted(available - known)]
    order = [snapshot for snapshot, _label in selected]
    labels = [label for _snapshot, label in selected]
    for name in ["LN", "Shaduf", "Horcrux", "Ballast-perch", "Ballast"]:
        vals = {r["snapshot"]: float(r["success_ratio"])
                for r in snap_rows if r["scheme"] == name}
        label = "Ballast-PC" if name == "Ballast-perch" else name
        style = STYLE["Ballast-PC"] if name == "Ballast-perch" else STYLE[name]
        ax.plot(range(len(order)), [vals[s] for s in order], label=label, **style)
    ax.set_xticks(range(len(order)), labels)
    ax.set_xlabel("Lightning topology snapshot")
    ax.grid(True, alpha=0.25)

    handles, labels_all = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_all, ncol=5, loc="upper center",
               frameon=False, fontsize=7.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = os.path.join(FIGURES, f"fig_robustness.{ext}")
        fig.savefig(out, dpi=220)
        print(f"wrote {out}")


def plot_epoch_tradeoff():
    """v8: pure epoch quotas and the optional receipted overflow tier."""
    epoch_rows = _read("epoch_sweep.csv")
    overflow_rows = _read("overflow_sweep.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))

    ax = axes[0]
    q = sorted((int(r["epoch_payments"]), float(r["success_ratio"]))
               for r in epoch_rows if r["scheme"] == "Ballast-Q")
    ax.plot([x for x, _ in q], [y for _, y in q], color="#0072B2",
            marker="o", lw=2.0, label="Ballast-Q")
    dynamic = next(float(r["success_ratio"]) for r in epoch_rows
                   if r["scheme"] == "Ballast-dynamic")
    perch = next(float(r["success_ratio"]) for r in epoch_rows
                 if r["scheme"] == "Ballast-PC")
    ax.axhline(dynamic, color="#009E73", ls="--", lw=1.2,
               label="dynamic upper bound")
    ax.axhline(perch, color="#777777", ls=":", lw=1.2,
               label="fixed per-channel")
    ax.set_xscale("log")
    ax.set_xlabel("payments per checkpoint epoch")
    ax.set_ylabel("payment success ratio")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, fontsize=7)

    ax = axes[1]
    xs = [float(r["overflow_fraction"]) for r in overflow_rows]
    success = [float(r["success_ratio"]) for r in overflow_rows]
    slow = [float(r["slow_path_rate_all_payments"]) for r in overflow_rows]
    line1 = ax.plot(xs, success, color="#0072B2", marker="o", lw=2.0,
                    label="success ratio")
    ax.set_xlabel("overflow reserve fraction $\\gamma$")
    ax.set_ylabel("payment success ratio", color="#0072B2")
    ax.tick_params(axis="y", labelcolor="#0072B2")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    line2 = ax2.plot(xs, slow, color="#d55e00", marker="s", ls="--", lw=1.6,
                     label="receipt slow-path rate")
    ax2.set_ylabel("fraction of all payments", color="#d55e00")
    ax2.tick_params(axis="y", labelcolor="#d55e00")
    ax.legend(line1 + line2, [x.get_label() for x in line1 + line2],
              frameon=False, fontsize=7, loc="lower right")

    fig.tight_layout()
    os.makedirs(FIGURES, exist_ok=True)
    for ext in ["pdf", "png"]:
        out = os.path.join(FIGURES, f"fig_epoch_tradeoff.{ext}")
        fig.savefig(out, dpi=220)
        print(f"wrote {out}")


def plot_theory_validation():
    """BALLAST-R E2: correlation degradation and finite-n tightness."""
    rows = _read("e2_correlation_scaling.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    styles = {
        "gamma-light-tail": dict(color="#0072B2", marker="o", lw=1.8),
        "lognormal-heavy-tail": dict(color="#D55E00", marker="s", lw=1.8, ls="--"),
    }
    labels = {"gamma-light-tail": "Gamma (light tail)",
              "lognormal-heavy-tail": "lognormal (heavy tail)"}
    for distribution in styles:
        unique = {}
        gain = {}
        for row in rows:
            if row["distribution"] != distribution:
                continue
            rho = float(row["rho"])
            unique[rho] = float(row["loglog_margin_slope"])
            if int(row["n"]) == 64:
                gain[rho] = float(row["multiplexing_gain"])
        xs = sorted(unique)
        axes[0].plot(xs, [unique[x] for x in xs], label=labels[distribution],
                     **styles[distribution])
        axes[1].plot(xs, [gain[x] for x in xs], label=labels[distribution],
                     **styles[distribution])
    axes[0].axhline(.5, color="#555555", ls=":", lw=1, label="$\\sqrt{n}$ theory")
    axes[0].set_xlabel("pairwise correlation $\\rho$")
    axes[0].set_ylabel("fitted margin exponent")
    axes[1].set_xlabel("pairwise correlation $\\rho$")
    axes[1].set_ylabel("multiplexing gain at $n=64$")
    for ax in axes:
        ax.grid(True, alpha=.25)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES, f"fig_theory_validation.{ext}")
        fig.savefig(out, dpi=220)
        print(f"wrote {out}")


def plot_safety_modes():
    """BALLAST-R E3: analytical fork bounds and fast-path value share."""
    rows = _read("e3_safety_grid.csv")
    finite = [r for r in rows if r["theta_quantile"] == "p95"
              and r["tau_e_s"] not in {"0", "inf"}
              and float(r["tau_e_s"]) >= 1.0]
    finite.sort(key=lambda r: float(r["tau_e_s"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    xs = [float(r["tau_e_s"]) for r in finite]
    axes[0].plot(xs, [float(r["theorem_upper_bound"]) for r in finite],
                 color="#0072B2", marker="o", lw=1.8, label="theorem upper bound")
    axes[0].plot(xs, [float(r.get("simulated_attack_exposure",
                                  r.get("analytical_attack_lower_bound"))) for r in finite],
                 color="#D55E00", marker="s", ls="--", lw=1.6,
                 label="analytical lower construction")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("checkpoint epoch $\\tau_e$ (s)")
    axes[0].set_ylabel("fork exposure (sat)")
    axes[0].legend(frameon=False, fontsize=7)

    one_tau = {r["theta_quantile"]: r for r in rows if r["tau_e_s"] == "10"}
    order = ["zero", "p50", "p80", "p95", "p99"]
    labels = ["0", "p50", "p80", "p95", "p99"]
    exposure_m = [float(one_tau[q]["theorem_upper_bound"]) / 1e6 for q in order]
    axes[1].plot(labels, exposure_m, color="#D55E00", marker="s", lw=1.8,
                 label="exposure bound")
    axes[1].set_xlabel("fast-path value cap $\\theta$")
    axes[1].set_ylabel("exposure bound (M sat)", color="#D55E00")
    axes[1].set_title("$\\tau_e=10$ s", fontsize=9)
    axes[1].tick_params(axis="y", labelcolor="#D55E00")
    coverage = axes[1].twinx()
    coverage.plot(labels, [float(one_tau[q]["fast_count_fraction"]) for q in order],
                  color="#0072B2", marker="o", lw=1.6, label="fast-path draws")
    coverage.plot(labels, [float(one_tau[q]["fast_value_fraction"]) for q in order],
                  color="#009E73", marker="^", ls="--", lw=1.5,
                  label="fast-path value")
    coverage.set_ylabel("fast-path fraction")
    coverage.set_ylim(0, 1.02)
    lines = axes[1].lines + coverage.lines
    coverage.legend(lines, [line.get_label() for line in lines],
                    frameon=False, fontsize=6, loc="upper left")
    for ax in axes:
        ax.grid(True, alpha=.25)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES, f"fig_safety_modes.{ext}")
        fig.savefig(out, dpi=220)
        print(f"wrote {out}")


def plot_claim_semantics():
    """BALLAST-R E4: availability blast radius of freeze versus escrow."""
    rows = [r for r in _read("e4_claim_semantics.csv")
            if abs(float(r["claim_rate_h"]) - .25) < 1e-9
            and abs(float(r["deposit_fraction"]) - .25) < 1e-9]
    rows.sort(key=lambda r: int(r["n_channels"]))
    fig, ax = plt.subplots(figsize=(4.2, 2.9))
    xs = [int(r["n_channels"]) for r in rows]
    ax.plot(xs, [float(r["freeze_refusal_rate"]) for r in rows],
            color="#D55E00", marker="s", lw=1.8, label="session freeze")
    ax.plot(xs, [float(r["escrow_refusal_rate"]) for r in rows],
            color="#0072B2", marker="o", lw=1.8, label="amount escrow")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("operator channel count $n$")
    ax.set_ylabel("honest draw refusal rate")
    ax.grid(True, alpha=.25)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = os.path.join(FIGURES, f"fig_claim_semantics.{ext}")
        fig.savefig(out, dpi=220)
        print(f"wrote {out}")


def emit_tab_perf():
    """LaTeX body for the paper's matched-capital table at capacity 4."""
    rows = _read("e1_pooling.csv")
    cap = 4
    sub = {("Ballast-PC" if r["scheme"] == "Ballast-perch" else r["scheme"]): r
           for r in rows if int(r["capacity"]) == cap}
    order = ["LN", "Revive", "Ballast-PC", "Shaduf", "Horcrux", "Ballast"]
    attrs = {
        "LN": ("no", "no"), "Revive": ("yes", "no"),
        "Ballast-PC": ("no", "no"), "Shaduf": ("yes", "no"),
        "Horcrux": ("yes", "$(\\checkmark)$"), "Ballast": ("no", "yes"),
    }
    labels = {
        "LN": "\\textsc{LN}", "Revive": "\\textsc{Revive}",
        "Ballast-PC": "\\textsc{Ballast-PC}", "Shaduf": "\\textsc{Shaduf}",
        "Horcrux": "\\textsc{Horcrux}", "Ballast": "\\ballast",
    }
    ln = float(sub["LN"]["success_ratio"])
    print(f"\n% --- tab:perf body (capacity scale {cap}, L=0) ---")
    for name in order:
        r = sub[name]
        sr = 100 * float(r["success_ratio"])
        delta = 100 * (float(r["success_ratio"]) - ln)
        delta_s = "---" if name == "LN" else f"${delta:+.1f}$"
        moves, pooled = attrs[name]
        print(f"{labels[name]:20s} & ${sr:.1f}\\%$ & {delta_s} & "
              f"{moves} & {pooled} \\\\")


def emit_tab_theta():
    rows = _read("q3_theta.csv")
    print("\n% --- Q3 capital efficiency / held-out exceedance ---")
    for r in rows:
        a = float(r["alpha"])
        g = float(r["mean_g"])
        ref = float(r["refusal_rate"])
        print(f"${a:.2f}$ & ${g:.2f}\\times$ & {ref:.3f} & $0$ \\\\")


if __name__ == "__main__":
    plot_latency()
    plot_q1()
    plot_robustness()
    plot_epoch_tradeoff()
    plot_theory_validation()
    plot_safety_modes()
    plot_claim_semantics()
    emit_tab_perf()
    emit_tab_theta()
