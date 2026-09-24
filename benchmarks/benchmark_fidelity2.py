#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
通用 fidelity 工具: 任意 workload 的 MPS 近似态 vs 精确态
用法: python benchmark_fidelity2.py <workload> <qubits> <depth> <bond>
输出: 打印 fidelity, 并追加保存 /root/results/fidelity_<name>.json
"""
import json
import sys
import time

import numpy as np
import qibo
from qibo import gates
from qibo import models as qibo_models


def build(workload, nqubits, depth):
    if workload == "QFT":
        return qibo_models.QFT(nqubits)
    rng = np.random.RandomState(2026)
    from qibo import Circuit
    c = Circuit(nqubits)
    for layer in range(depth):
        for i in range(nqubits):
            if rng.random() > 0.5:
                c.add(gates.T(i))
            else:
                c.add(gates.RX(i, rng.uniform(0, 3.14)))
        if layer % 2 == 0:
            pairs = [(i, i + 1) for i in range(0, nqubits - 1, 2)]
        else:
            pairs = [(i, i + 1) for i in range(1, nqubits - 1, 2)]
        for a, b in pairs:
            c.add(gates.CZ(a, b))
    return c


def run(workload, nqubits, depth, mode, bond):
    mps_opts = {"max_bond": bond} if mode == "mps" else None
    qibo.set_backend(backend="qibotn", platform="qutensornet", runcard={
        "MPI_enabled": False,
        "MPS_enabled": mps_opts is not None,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    })
    if mps_opts is not None:
        qibo.get_backend().mps_opts = mps_opts
    c = build(workload, nqubits, depth)
    t0 = time.perf_counter()
    state = np.asarray(c().state())
    return state, time.perf_counter() - t0


def main():
    workload, nqubits, depth, bond = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    print(f"=== fidelity: {workload}({nqubits},d{depth}) exact vs MPS(bond={bond}) ===", flush=True)

    exact, t1 = run(workload, nqubits, depth, "dense", 0)
    print(f"[exact] {t1:.2f}s", flush=True)

    approx, t2 = run(workload, nqubits, depth, "mps", bond)
    overlap = np.vdot(exact, approx)
    fid = float(abs(overlap) ** 2 / (np.linalg.norm(exact) ** 2 * np.linalg.norm(approx) ** 2))
    max_err = float(np.max(np.abs(exact - approx)))
    print(f"[mps bond={bond}] {t2:.2f}s fidelity={fid:.10f} max_abs_err={max_err:.3e}", flush=True)

    out = f"/root/results/fidelity_{workload}_{nqubits}_d{depth}_b{bond}.json"
    with open(out, "w") as f:
        json.dump({"workload": workload, "qubits": nqubits, "depth": depth,
                   "bond": bond, "exact_runtime": round(t1, 3),
                   "mps_runtime": round(t2, 3), "fidelity": fid,
                   "max_abs_err": max_err}, f, indent=2)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
