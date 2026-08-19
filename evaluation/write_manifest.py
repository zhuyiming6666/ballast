"""Write an auditable manifest for the checked-in BALLAST-R results."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import platform
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_files(directory, suffixes):
    rows = {}
    for base, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in {"node_modules", "artifacts", "cache", "__pycache__", ".venv"}]
        for name in files:
            if os.path.splitext(name)[1] not in suffixes:
                continue
            path = os.path.join(base, name)
            rows[os.path.relpath(path, ROOT)] = sha256(path)
    return dict(sorted(rows.items()))


def main():
    inputs = [
        os.path.join(HERE, "payment_value", "payment_value_satoshi_03.csv"),
        os.path.join(HERE, "network", "lightning_simplified_component.edgelist"),
        os.path.join(HERE, "network", "snapshots", "ln_600000.edgelist"),
        os.path.join(HERE, "network", "snapshots", "ln_640000.edgelist"),
        os.path.join(HERE, "network", "snapshots", "ln_677167.edgelist"),
        os.path.join(HERE, "network", "snapshots", "ln_20260804_rgs.edgelist"),
        os.path.join(ROOT, "datasets", "20230716.gml.geo"),
        os.path.join(ROOT, "datasets", "geo_snapshot_20230716_metadata.json"),
        os.path.join(ROOT, "datasets", "rgs_snapshot_20260804_v2.bin"),
        os.path.join(ROOT, "datasets", "rgs_snapshot_20260804_metadata.json"),
    ]
    packages = {}
    for name in ("networkx", "numpy", "scipy", "matplotlib", "coincurve"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    result_hashes = relative_files(
        os.path.join(ROOT, "results"), {".csv", ".json"})
    result_hashes.pop(os.path.join("results", "run_manifest.json"), None)
    manifest = {
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "reported_configuration": {
            "main": {"seeds": 10, "payments_per_seed": 50000,
                     "capacity_scales": list(range(1, 26)), "phi": 0.30},
            "snapshots": {"seeds": 10, "payments_per_seed": 50000,
                          "capacity_scale": 4, "phi": 0.30},
            "dataset_regimes": {"distance_samples_per_snapshot": 256,
                                  "seed": 20260808,
                                  "2023_capacity_experiment": False},
            "theory": {"seed": 20260803, "samples": 40000,
                       "payment_values": 200000, "claim_seeds_per_cell": 20,
                       "claim_horizon_h": 24, "checkpoint_gas": 87645,
                       "theta_endpoints": ["zero", "p50", "p80", "p95", "p99", "inf"],
                       "admission": {"counterparties": 8,
                                     "inflight_per_counterparty": 1,
                                     "r_min_s": 0.2, "delta_s": 0.2}},
            "sizing": {"seeds": 3, "payments_per_seed": 20000,
                       "max_trace": 500000},
            "measured_correlation": {"seeds": 10,
                                     "payments_per_seed": 50000,
                                     "bins": 100},
            "route_workload_sensitivity": {"seeds": 3,
                                             "payments_per_seed": 2000,
                                             "routes": ["shortest", "k3"],
                                             "workloads": ["uniform", "pareto80", "cyclic"]},
        },
        "input_sha256": {
            os.path.relpath(path, ROOT): sha256(path)
            for path in inputs if os.path.exists(path)
        },
        "source_sha256": relative_files(
            ROOT, {".py", ".sol", ".circom", ".js", ".sh"}),
        "result_sha256": result_hashes,
        "adaptive_contract_result_sha256": relative_files(
            os.path.join(ROOT, "onchain-adaptive", "results"), {".csv", ".json"}),
    }
    target = os.path.join(ROOT, "results", "run_manifest.json")
    with open(target, "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
