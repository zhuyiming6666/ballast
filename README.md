# BALLAST: Node-Level Logical Liquidity for Payment Channel Networks

This repository contains the simulation code, Lightning Network inputs, evaluation drivers, result tables, plotting utilities, and EVM prototype used to evaluate **BALLAST**. BALLAST pools a cooperatively provisioned node-level bond and uses it to issue logical draws to individual channels. Bond provisioning happens once when a BALLAST session is established; subsequent logical draws are unilateral on the fast path.

The artifact supports three main tasks:

- reproducing the payment-success, pooling, sizing, security, and sensitivity experiments;
- regenerating the CSV tables and figures used by the paper; and
- compiling, testing, and benchmarking the adaptive Solidity contract.

## Table of Contents

- [Artifact Layout](#artifact-layout)
- [Automated Reproduction](#automated-reproduction)
- [Manual Reproduction](#manual-reproduction)
- [Running Guide](#running-guide)
- [Parameter Description](#parameter-description)
- [Outputs](#outputs)
- [System Requirements](#system-requirements)

## Artifact Layout

| Path | Description |
|---|---|
| `evaluation/` | Python simulator, experiment drivers, tests, and plotting utilities |
| `evaluation/network/` | Baseline and temporal Lightning Network snapshots |
| `evaluation/payment_value/` | March 2021 transaction-value trace used as the payment-amount proxy |
| `results/` | Checked-in and regenerated experiment results in CSV/JSON form |
| `onchain-adaptive/contracts/` | Solidity implementation of the adaptive checkpoint, confirmation, and claim-escrow contract |
| `onchain-adaptive/test/` | Hardhat adversarial and functional tests |
| `onchain-adaptive/scripts/` | Gas, four-node, WAN, and Sepolia benchmark drivers |
| `onchain-adaptive/results/` | Contract benchmark outputs |

The simulator compares LN, Revive, Shaduf, Horcrux, BALLAST, and the per-channel BALLAST-PC ablation under shared topologies, balances, payment sequences, and random seeds. The main workload uses `evaluation/network/lightning_simplified_component.edgelist` and `evaluation/payment_value/payment_value_satoshi_03.csv`.

## Automated Reproduction

### Full Paper Workflow

`evaluation/run_all_paper.sh` locates a working Python interpreter, prepares dependencies when needed, executes the paper-scale experiments, regenerates figures, and writes a result manifest.

#### Usage

```bash
# Run from the artifact root.
cd evaluation

# Full paper configuration using all available CPU cores.
bash run_all_paper.sh

# Limit parallel workers on RAM-constrained machines.
JOBS=8 bash run_all_paper.sh

# Run in the background and retain a log.
nohup bash run_all_paper.sh > run_all.log 2>&1 &
tail -f run_all.log
```

The full workflow is compute intensive. Intermediate and final tables are written to `results/`; existing files with the same names are replaced.

### Dataset-Audit Prerequisite

The first stage of `run_all_paper.sh` performs the 2021/2023/2026 dataset-regime audit. In addition to the network files bundled under `evaluation/network/`, that stage expects the following auxiliary files under `datasets/`:

```text
datasets/
├── 20230716.gml.geo
├── geo_snapshot_20230716_metadata.json
├── rgs_channels_20260804.csv
├── rgs_snapshot_20260804_metadata.json
└── rgs_snapshot_20260804_v2.bin
```

These files are used only for the topology/policy audit and provenance manifest. The bundled 2021 and 2026 edgelists and the bundled payment-value trace are sufficient for the core capacity-constrained simulations described below.

### Post-Run Check

```bash
# From the artifact root.
test -f results/e1_pooling.csv
test -f results/e_latency.csv
test -f results/q2_sensitivity.csv
test -f results/q3_theta.csv
test -f results/run_manifest.json
```

## Manual Reproduction

### Install Python Dependencies

```bash
# Run from the artifact root.
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r evaluation/requirements.txt
```

The Python requirements are pinned by compatible version ranges in `evaluation/requirements.txt`: NetworkX, NumPy, SciPy, Matplotlib, and coincurve.

### Run the Unit Tests

```bash
python3 -m unittest discover -s evaluation -p 'test_*.py'
```

The test suite covers admission control and the shared scheme implementations. It should complete in a few seconds and does not run the paper-scale Monte Carlo experiments.

### Run a Small Simulation

```bash
# Small Q1 payment-performance run.
python3 evaluation/run_experiments.py \
  --which q1 \
  --tx_load 2000 \
  --repeat 2 \
  --capacities 1 2 4

# Closed-form security and economics checks.
python3 evaluation/run_security_sensitivity.py
python3 evaluation/run_deterrence_economics.py
python3 evaluation/run_pooling_effectiveness.py
```

### Compile and Test the Contract

```bash
cd onchain-adaptive
npm ci
npm run compile
npm test
npm run bench
```

`npm run bench` writes `onchain-adaptive/results/adaptive_gas.csv`. The Hardhat network uses the Cancun hard fork, Solidity 0.8.24, and the optimizer with 200 runs.

## Running Guide

### 1. Payment Performance and Pooling

```bash
# Main Q1/Q2/Q3 driver with lightweight defaults.
python3 evaluation/run_experiments.py --which all

# Paper-scale pooling attribution and direct-latency table.
python3 evaluation/run_ablation.py --paper --jobs 8

# Equal-capital skim-fraction sensitivity.
python3 evaluation/run_phi_sweep.py --paper --jobs 8

# Cross-snapshot robustness.
python3 evaluation/run_snapshot_robustness.py --paper --jobs 8
```

### 2. Sizing, Security, and Robustness

```bash
# Online sizing closure and capital utilization.
python3 evaluation/run_sizing_closure.py \
  --tx-load 20000 --repeat 3 --max-trace 500000
python3 evaluation/run_capital_utilization.py --paper

# Fork-containment, claim semantics, and theory experiments.
python3 evaluation/run_security_sensitivity.py
python3 evaluation/run_theory_experiments.py \
  --samples 40000 --max-values 200000 \
  --claim-seeds 20 --claim-horizon 24

# Correlated demand and throttled-hub stress tests.
python3 evaluation/run_correlated_shock.py
python3 evaluation/run_throttled_hub.py
```

### 3. Route and Workload Sensitivity

```bash
python3 evaluation/run_route_workload_sensitivity.py \
  --tx-load 2000 --repeat 3

python3 evaluation/run_measured_correlation.py \
  --tx-load 50000 --repeat 10 --bins 100
```

### 4. Contract Benchmarks

```bash
cd onchain-adaptive

# Local gas benchmark.
npx hardhat run scripts/bench_gas.js

# Four-account end-to-end benchmark.
npx hardhat run scripts/bench_4node.js

# Public Sepolia benchmark after placing a funded test-only private key in
# ~/.ballast_sepolia_key with owner-only permissions.
SEPOLIA_RPC="https://your-sepolia-rpc.example" node scripts/bench_sepolia.js 10
```

The Sepolia benchmark deploys a contract and spends testnet ETH. Never place a mainnet key in `~/.ballast_sepolia_key`. The WAN benchmark uses local JSON-RPC endpoints and `scripts/delay_proxy.py`; inspect `scripts/bench_wan.js` for the expected ports before running it. The fixed private keys in that script are Hardhat development keys and must never be used with real funds.

### 5. Regenerate Figures and Manifest

```bash
# Run after the required result CSVs have been generated.
python3 evaluation/plot_results.py
python3 evaluation/write_manifest.py
```

## Parameter Description

Command-line names are not completely uniform across legacy drivers: some scripts use underscores (for example, `--tx_load`) and newer scripts use hyphens (`--tx-load`). Run `python3 evaluation/<script>.py --help` for the exact interface.

| Parameter | Meaning | Typical paper value |
|---|---|---|
| `--paper` | Select the paper-scale configuration instead of lightweight defaults | enabled for final tables |
| `--tx_load` / `--tx-load` | Number of attempted payments per seed | 50,000 for main topology points |
| `--repeat` | Number of independent seeds | 10 for main topology points |
| `--phi` | Fraction of channel balance skimmed into the node bond | 0.30 |
| `--cap` | Capacity scaling factor for the selected snapshot | 4 in several sensitivity runs |
| `--capacities` | List of capacity scales to sweep | experiment specific |
| `--max_trace` / `--max-trace` | Maximum number of payment values parsed from the trace | 200,000 or 500,000 |
| `--jobs` | Parallel worker count | all CPU cores by default |
| `--alpha` | Target service level for adaptive sizing | 0.99 |
| `--lambda_r` / `--lambda-r` | Coordination-cost weight | 0.05 |
| `--theta` | Fast-path amount threshold used by security sweeps | derived from the trace when omitted |
| `--samples` | Monte Carlo samples for the theory driver | 40,000 |
| `--claim-seeds` | Independent seeds for claim simulations | 20 |
| `--claim-horizon` | Claim-simulation horizon in hours | 24 |

## Outputs

### Simulation Results

| Output | Produced by | Purpose |
|---|---|---|
| `results/e1_pooling.csv` | `run_ablation.py` | Equal-capital pooling attribution |
| `results/e_latency.csv` | `run_ablation.py` | Direct coordination-latency table |
| `results/q1_perf.csv` | `run_experiments.py` | Payment success at zero coordination latency |
| `results/q2_sensitivity.csv` | `run_sensitivity.py` | Coordination-cost sensitivity band |
| `results/q3_theta.csv` | `run_experiments.py` | Capital efficiency versus service level |
| `results/phi_sweep.csv` | `run_phi_sweep.py` | Bond skim-fraction sensitivity |
| `results/snapshot_robustness.csv` | `run_snapshot_robustness.py` | Temporal snapshot robustness |
| `results/epoch_sweep.csv` | `run_epoch_sweep.py` | Checkpoint-epoch sensitivity |
| `results/overflow_sweep.csv` | `run_overflow_sweep.py` | Optional overflow slow-path sensitivity |
| `results/e2_sizing_closure.csv` | `run_sizing_closure.py` | Adaptive sizing closure |
| `results/e3_safety_grid.csv` | `run_theory_experiments.py` | Fork-containment safety grid |
| `results/e4_claim_semantics.csv` | `run_theory_experiments.py` | Claim-semantics experiment |
| `results/run_manifest.json` | `write_manifest.py` | Environment, input hashes, source hashes, and result hashes |

Additional checked-in CSVs under `results/` contain the deployment, security, margin-attribution, mixed-workload, micro-payment, reserve, and stress-test analyses.

### Contract Results

| Output | Produced by |
|---|---|
| `onchain-adaptive/results/adaptive_gas.csv` | `scripts/bench_gas.js` |
| `onchain-adaptive/results/four_node_e2e.csv` | `scripts/bench_4node.js` |
| `onchain-adaptive/results/wan_latency.csv` | `scripts/bench_wan.js` |
| `onchain-adaptive/results/sepolia_latency.csv` | `scripts/bench_sepolia.js` |

## System Requirements

### Hardware Requirements

- No GPU, trusted execution environment, or specialized hardware is required.
- A multi-core CPU is recommended for paper-scale Monte Carlo sweeps.
- Full runs may be RAM intensive; set `JOBS` or `--jobs` to limit parallelism.
- Sufficient disk space is required for the 70 MB payment-value trace and generated result files.

### Software Requirements

| Component | Requirement |
|---|---|
| Operating system | Linux or macOS; the experiment code is platform independent Python |
| Shell | Bash for `evaluation/run_all_paper.sh` |
| Python | Python 3.10--3.12 recommended |
| Python packages | NetworkX, NumPy, SciPy, Matplotlib, coincurve |
| Node.js | Node.js 22 recommended for the evaluated contract environment |
| Package manager | npm |
| Solidity toolchain | Hardhat 2.28.x with Solidity 0.8.24 |

Internet access is needed only when dependencies must be installed, when auxiliary datasets are obtained, or when the Sepolia benchmark is run. The bundled simulator inputs and local Hardhat tests run offline after dependencies are installed.
