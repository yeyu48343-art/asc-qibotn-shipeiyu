#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化3: MPS (Matrix Product State) 近似模拟
通过 qibotn 后端的 mps_opts 参数, 切换到 CircuitMPS 模式,
用截断键维数 (max_bond) 换取速度, 对比精确收缩 vs MPS 近似。

原理: 精确收缩(默认)把整个张量网络精确求值, 复杂度随 qubit 指数增长。
MPS 近似把量子态表示为矩阵乘积链, 复杂度随 max_bond 多项式增长,
代价是引入截断误差 (bond dimension 越小越快但越不准)。
"""

import os
import csv
import time
import gc
import qibo
from qibo import gates
from qibo import models as qibo_models
import benchmark as bm


def init_backend_with_mps(max_bond):
    """初始化后端, 启用 MPS 近似, 指定最大键维数"""
    computation_settings = {
        "MPI_enabled": False,
        "MPS_enabled": True,    # 关键: 启用 MPS
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    qibo.set_backend(backend="qibotn", platform="qutensornet",
                     runcard=computation_settings)
    # 配置 MPS 的键维数
    b = qibo.get_backend()
    # mps_opts 是传给 quimb CircuitMPS 的 gate_opts
    b.mps_opts = {"max_bond": max_bond} if max_bond > 0 else None
    mode = f"MPS(max_bond={max_bond})" if max_bond > 0 else "dense(精确)"
    print(f"[INFO] 后端模式: {mode}")


def run_single_mps(workload, nqubits, depth, max_bond):
    """用 MPS 模式跑单个线路"""
    builder = {
        "QFT": lambda n: qibo_models.QFT(n),
        "Supremacy": lambda n: bm.build_supremacy(n, depth),
    }[workload]
    c = builder(nqubits)
    ngates = c.ngates

    label = f"MPS(bond={max_bond})" if max_bond > 0 else "dense"
    print(f"  {workload}({nqubits}) {label}, {ngates} 门, 收缩...", end="", flush=True)
    t0 = time.perf_counter()
    try:
        result = c()
        runtime = time.perf_counter() - t0
        status = "success"
        sv = result.state()
        print(f" 完成: {runtime:.2f}s [{status}, |ψ|形状={sv.shape}]")
    except Exception as e:
        runtime = time.perf_counter() - t0
        status = f"FAIL:{type(e).__name__}"
        print(f" 失败: {runtime:.2f}s [{status}]")
    return runtime, status, ngates


def main():
    print("=== 优化3: MPS 近似 vs 精确收缩 ===")
    print(f"threads: OMP={os.environ.get('OMP_NUM_THREADS','未设')} (保持 baseline 4 线程)")
    print()

    rows = []
    threads_info = f"OMP={os.environ.get('OMP_NUM_THREADS','default')}"

    # 先精确 (max_bond=0 表示不截断, 作为对照)
    # 再 MPS 近似: bond=64, 128, 256
    configs = [
        (0, "精确收缩"),     # 对照, 等价于 baseline
        (64, "MPS bond=64 (激进截断, 最快最不准)"),
        (128, "MPS bond=128 (中等)"),
        (256, "MPS bond=256 (保守, 慢但较准)"),
    ]

    for max_bond, desc in configs:
        print(f"\n  --- {desc} ---")
        # 每次重新初始化后端 (切换 MPS 配置)
        init_backend_with_mps(max_bond)
        for wl, depth in [("QFT", 4), ("Supremacy", 8)]:
            nq = 28
            runtime, status, ngates = run_single_mps(wl, nq, depth, max_bond)
            label = f"mps_bond{max_bond}" if max_bond > 0 else "dense_ref"
            rows.append({
                "tag": f"opt3_{label}",
                "workload": wl,
                "qubits": nq,
                "ngates": ngates,
                "command": f"python benchmark_mps.py [max_bond={max_bond} + {threads_info}]",
                "runtime": f"{runtime:.4f}",
                "status": status,
            })
            gc.collect()

    with open("results.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag", "workload", "qubits", "ngates",
            "command", "runtime", "status"
        ])
        for r in rows:
            writer.writerow(r)

    print(f"\n=== 优化3 完成 ===")
    print(f"{'workload':<12} {'mode':<18} {'runtime':<12} {'status'}")
    print("-" * 60)
    for r in rows:
        mode = r["tag"].replace("opt3_", "")
        print(f"{r['workload']:<12} {mode:<18} {r['runtime']:<12} {r['status']}")


if __name__ == "__main__":
    main()
