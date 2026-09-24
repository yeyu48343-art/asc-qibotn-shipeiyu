#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPS 近似的 fidelity 量化: 在 24 qubits 上对比 MPS 近似态与精确态

fidelity = |<psi_exact | psi_mps>|^2  (两个归一化态的内积模方, 1=完全一致, 越低失真越大)

24 qubits 态向量 = 268 MB, 本机可同时容纳两个。回答评审"正确性验证不足"的质疑:
MPS 的截断误差不再只是定性描述, 而是给出定量数值。
"""
import json
import time

import numpy as np
import qibo
from qibo import models as qibo_models


def run_qft24(mode, bond):
    mps_opts = {"max_bond": bond} if mode == "mps" else None
    qibo.set_backend(backend="qibotn", platform="qutensornet", runcard={
        "MPI_enabled": False,
        "MPS_enabled": mps_opts is not None,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    })
    if mps_opts is not None:
        qibo.get_backend().mps_opts = mps_opts
    c = qibo_models.QFT(24)
    t0 = time.perf_counter()
    state = np.asarray(c().state())
    runtime = time.perf_counter() - t0
    return state, runtime


def main():
    print("=== MPS fidelity 量化 (QFT 24q) ===", flush=True)

    exact, t_exact = run_qft24("dense", 0)
    exact_norm = np.linalg.norm(exact)
    print(f"[exact] runtime={t_exact:.2f}s norm={exact_norm:.6f}", flush=True)

    results = {"exact_runtime": round(t_exact, 3), "mps": {}}
    for bond in [64, 128, 256]:
        approx, t = run_qft24("mps", bond)
        # fidelity = |<exact|approx>|^2 (各自归一化)
        overlap = np.vdot(exact, approx)
        fid = float(abs(overlap) ** 2 / (exact_norm ** 2 * np.linalg.norm(approx) ** 2))
        max_err = float(np.max(np.abs(exact - approx)))
        print(f"[mps bond={bond}] runtime={t:.2f}s fidelity={fid:.8f} "
              f"max_abs_err={max_err:.3e}", flush=True)
        results["mps"][bond] = {
            "runtime": round(t, 3), "fidelity": round(fid, 8),
            "max_abs_err": max_err}

    results["exact_dim"] = int(exact.shape[0])
    out = "/home/longm/asc-qibotn-new/results/fidelity_qft24.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved: {out}", flush=True)


if __name__ == "__main__":
    main()
