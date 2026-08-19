#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_all_paper.sh -- auto-install deps, then regenerate all paper tables/figures.
#
#   results/e1_pooling.csv     pooling-attribution sweep (cap 1..25)  [heavy]
#   results/e_latency.csv      direct rebalance-latency table          [heavy]
#   results/q2_sensitivity.csv coordination-cost band for fig:latency  [light]
#   results/q3_theta.csv       out-of-sample capital-efficiency table  [light]
#   results/phi_sweep.csv      equal-capital phi sensitivity           [heavy]
#   results/snapshot_robustness.csv temporal snapshot robustness       [heavy]
#   results/dataset_regimes.csv  2021/2023/2026 field/regime audit     [light]
#   results/epoch_sweep.csv      v8 checkpoint-amortization sweep      [heavy]
#   results/overflow_sweep.csv   v8 optional receipt slow-path sweep   [heavy]
#
# The setup stage is self-healing: it probes several Python interpreters,
# SKIPS any that the OS kills (macOS "Killed: 9" / broken signature), builds a
# venv when it can, and falls back to --user / --break-system-packages, or even
# `brew install python` if no working interpreter exists.
#
# Usage (from anywhere):
#   bash run_all_paper.sh            # auto-setup + full paper config, all cores
#   JOBS=8 bash run_all_paper.sh     # cap worker count (RAM-bound boxes)
#
# Detached, survives terminal close, logs to run_all.log:
#   nohup bash run_all_paper.sh > run_all.log 2>&1 &
#   tail -f run_all.log
# ---------------------------------------------------------------------------
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/ballast-matplotlib"
mkdir -p "$MPLCONFIGDIR"

DEPS="networkx numpy scipy matplotlib"

# --- does this interpreter actually run (not SIGKILLed) and import deps? ----
py_runs()  { "$1" -c 'print("ok")'                  >/dev/null 2>&1; }
py_deps()  { "$1" -c 'import networkx,numpy,scipy,matplotlib' >/dev/null 2>&1; }

# --- first interpreter that EXECUTES (broken/Killed:9 ones are skipped) -----
find_python() {
  local c
  for c in .venv/bin/python3 \
           python3.12 python3.11 python3.10 python3 \
           /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
           /opt/homebrew/bin/python3 \
           /usr/local/bin/python3.12 /usr/local/bin/python3 \
           /usr/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && py_runs "$c"; then
      command -v "$c"; return 0
    fi
  done
  return 1
}

# --- last resort: install a fresh Python via Homebrew (macOS) ---------------
brew_install_python() {
  local brew=""
  for b in brew /opt/homebrew/bin/brew /usr/local/bin/brew; do
    command -v "$b" >/dev/null 2>&1 && { brew="$b"; break; }
  done
  [ -z "$brew" ] && return 1
  echo "[setup] installing python@3.12 via Homebrew (this can take a minute)..."
  "$brew" install python@3.12 >/dev/null 2>&1 || return 1
  return 0
}

echo "==== [setup] locating a working Python ============================="
PY="$(find_python || true)"
if [ -z "${PY:-}" ]; then
  echo "[setup] no usable Python found -- trying Homebrew"
  brew_install_python && PY="$(find_python || true)"
fi
if [ -z "${PY:-}" ]; then
  echo "[setup] FATAL: no Python interpreter would run on this machine."
  echo "        Install one (e.g. 'brew install python@3.12') and re-run."
  exit 1
fi
echo "[setup] python: $PY  ($($PY --version 2>&1))"

# --- ensure the three deps are importable, trying isolated -> user -> system
install_deps() {
  echo "[setup] installing deps: $DEPS"
  # 1) cleanest: a project-local venv (skip if it can't even be built)
  if [ ! -x .venv/bin/python3 ]; then
    if "$PY" -m venv .venv >/dev/null 2>&1 && [ -x .venv/bin/python3 ]; then
      echo "[setup] created .venv"
    else
      echo "[setup] venv unavailable -- will install against $PY directly"
    fi
  fi
  if [ -x .venv/bin/python3 ] && py_runs .venv/bin/python3; then
    PY=".venv/bin/python3"
    "$PY" -m pip install -q --upgrade pip       >/dev/null 2>&1 || true
    "$PY" -m pip install -q $DEPS               >/dev/null 2>&1 || true
  fi
  py_deps "$PY" && return 0
  # 2) per-user site-packages
  "$PY" -m pip install -q --user $DEPS          >/dev/null 2>&1 || true
  py_deps "$PY" && return 0
  # 3) externally-managed envs (PEP 668)
  "$PY" -m pip install -q --break-system-packages $DEPS >/dev/null 2>&1 || true
  py_deps "$PY" && return 0
  return 1
}

if py_deps "$PY"; then
  echo "[setup] deps already present"
else
  install_deps || {
    echo "[setup] FATAL: could not install $DEPS for $PY"
    echo "        Try manually: $PY -m pip install $DEPS"
    exit 1
  }
fi
echo "[setup] OK -> $PY  ($($PY -c 'import networkx,numpy,scipy,matplotlib;print("networkx",networkx.__version__,"numpy",numpy.__version__,"scipy",scipy.__version__,"matplotlib",matplotlib.__version__)'))"
echo "[setup] jobs: ${JOBS:-all cores}"

JOBS_ARG=""
[ -n "${JOBS:-}" ] && JOBS_ARG="--jobs ${JOBS}"

# ---------------------------------------------------------------------------
set -e
t0=$(date +%s)

echo; echo "==== [1/12] dataset regime audit (2021/2023/2026) =================="
$PY run_dataset_regimes.py

echo; echo "==== [2/12] pooling attribution + latency (heavy) ================="
$PY run_ablation.py --paper $JOBS_ARG

echo; echo "==== [3/12] coordination-cost sensitivity band (fig:latency) ======"
$PY run_sensitivity.py --paper

echo; echo "==== [4/12] out-of-sample capital-efficiency table (q3) ==========="
$PY run_experiments.py --which q3 --paper

echo; echo "==== [5/12] equal-capital phi sensitivity =========================="
$PY run_phi_sweep.py --paper $JOBS_ARG

echo; echo "==== [6/12] temporal-snapshot robustness ==========================="
$PY generate_snapshots.py
$PY run_snapshot_robustness.py --paper ${JOBS_ARG:-}

echo; echo "==== [7/12] route/workload sensitivity =============================="
$PY run_route_workload_sensitivity.py --tx-load 2000 --repeat 3

echo; echo "==== [8/12] measured cross-channel correlation ======================"
$PY run_measured_correlation.py --tx-load 50000 --repeat 10

echo; echo "==== [9/12] v8 epoch-checkpoint sweep ==============================="
$PY run_epoch_sweep.py --paper ${JOBS_ARG:-}

echo; echo "==== [10/12] v8 optional-overflow sweep ============================="
$PY run_overflow_sweep.py --paper ${JOBS_ARG:-}

echo; echo "==== [11/12] BALLAST-R theory experiments ==========================="
# Explicitly reproduce the configuration reported in the manuscript.  --paper
# remains available for a larger robustness run with different sample counts.
$PY run_theory_experiments.py --samples 40000 --max-values 200000 \
    --claim-seeds 20 --claim-horizon 24

echo; echo "==== [12/12] sizing closure + figures ================================"
$PY run_sizing_closure.py --tx-load 20000 --repeat 3 --max-trace 500000
$PY run_correlated_shock.py
$PY run_throttled_hub.py
$PY plot_results.py
$PY write_manifest.py

echo
echo "==== done in $(( $(date +%s) - t0 ))s -- tables in ../results/ ========"
ls -la ../results/e1_pooling.csv ../results/e_latency.csv \
       ../results/q2_sensitivity.csv ../results/q3_theta.csv \
       ../results/phi_sweep.csv ../results/snapshot_robustness.csv \
       ../results/epoch_sweep.csv ../results/overflow_sweep.csv \
       ../results/e2_correlation_scaling.csv ../results/e2_sizing_closure.csv \
       ../results/e3_safety_grid.csv ../results/e4_claim_semantics.csv
