#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单次实验执行器: 由 run_scale_suite.py 以子进程方式调用, 便于超时控制与内存隔离。

用法:
  python benchmark_scale.py <workload> <qubits> <depth> <mode> <bond>
  workload: QFT | Supremacy
  mode: dense | mps
  bond: 仅 mps 模式有效

输出: 单行 JSON {"workload":..,"qubits":..,"depth":..,"mode":..,"bond":..,
               "ngates":..,"runtime":..,"status":..,"detail":..}
"""
import json
import sys
import time

import qibo
from qibo import models as qibo_models


def build_supremacy(nqubits, depth):
    """与 benchmark.py 保持一致: 随机单比特门 + 交替 CZ, 每层约 1.5n 个门"""
    import numpy as np
    rng = np.random.RandomState(2026)
    c = qibo.Circuit(nqubits) if hasattr(qibo, "Circuit") else None
    from qibo import Circuit, gates
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


def main():
    workload, nqubits, depth, mode, bond = (
        sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4],
        int(sys.argv[5]) if len(sys.argv) > 5 else 0)

    mps_opts = None
    if mode == "mps":
        mps_opts = {"max_bond": bond}

    qibo.set_backend(backend="qibotn", platform="qutensornet", runcard={
        "MPI_enabled": False,
        "MPS_enabled": mps_opts is not None,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    })
    if mps_opts is not None:
        qibo.get_backend().mps_opts = mps_opts

    if workload == "QFT":
        c = qibo_models.QFT(nqubits)
    else:
        c = build_supremacy(nqubits, depth)

    ngates = c.ngates
    t0 = time.perf_counter()
    try:
        result = c()
        runtime = time.perf_counter() - t0
        sv = result.state()
        status = "success"
        detail = f"state_dim={sv.shape[0]}"
    except MemoryError as e:
        runtime = time.perf_counter() - t0
        status = "OOM"
        detail = f"MemoryError:{str(e)[:80]}"
    except Exception as e:
        runtime = time.perf_counter() - t0
        status = f"FAIL:{type(e).__name__}"
        detail = str(e)[:80]

    print(json.dumps({
        "workload": workload, "qubits": nqubits, "depth": depth,
        "mode": mode, "bond": bond, "ngates": ngates,
        "runtime": round(runtime, 4), "status": status, "detail": detail,
    }))


if __name__ == "__main__":
    main()
