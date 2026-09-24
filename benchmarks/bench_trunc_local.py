#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""本地截断证据: Supremacy 20q d30, 精确 vs MPS b16/b64/b256, 只打印不落盘"""
import time
import numpy as np
import qibo
from qibo import gates
from qibo import models as qibo_models


def build_supremacy(nqubits, depth):
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


def run(mode, bond):
    mps_opts = {"max_bond": bond} if mode == "mps" else None
    qibo.set_backend(backend="qibotn", platform="qutensornet", runcard={
        "MPI_enabled": False,
        "MPS_enabled": mps_opts is not None,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    })
    if mps_opts is not None:
        qibo.get_backend().mps_opts = mps_opts
    c = build_supremacy(20, 30)
    t0 = time.perf_counter()
    return np.asarray(c().state()), time.perf_counter() - t0


exact, t1 = run("dense", 0)
print(f"[exact] {t1:.2f}s dim={exact.shape[0]}", flush=True)
for bond in [16, 64, 256]:
    approx, t2 = run("mps", bond)
    overlap = np.vdot(exact, approx)
    fid = float(abs(overlap) ** 2 / (np.linalg.norm(exact) ** 2 * np.linalg.norm(approx) ** 2))
    print(f"[mps b={bond}] {t2:.2f}s fidelity={fid:.10f}", flush=True)
