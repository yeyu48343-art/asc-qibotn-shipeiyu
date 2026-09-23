#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASC 选拔作业 - 大题 QiboTN benchmark 脚本
跑 QFT / QAOA / Supremacy 三种量子线路, 记录 runtime 和 status, 输出 results.csv

用法:
    python benchmark.py [--workloads QFT,QAOA,Supremacy] [--qubits 8,10,12] [--tag baseline]
"""

import argparse
import csv
import os
import sys
import time
import gc

# 必须在 import qibo 前设线程数 (BLAS/cotengra 用)
# 通过环境变量传入, 这里不硬编码
import qibo
from qibo import Circuit, gates
from qibo import models as qibo_models


def build_qft(nqubits):
    """构造 QFT 线路 (Quantum Fourier Transform)"""
    return qibo_models.QFT(nqubits)


def build_qaoa(nqubits):
    """构造 QAOA 前向线路 (固定参数, 不做优化, 只测一次前向计算)"""
    import numpy as np
    c = Circuit(nqubits)
    # 初始化: 均匀叠加
    for i in range(nqubits):
        c.add(gates.H(i))
    # 一层 cost + mixer (固定参数, 典型 QAOA p=1)
    np.random.seed(42)
    gamma = 0.5
    beta = 0.3
    # Cost: ZZ 相互作用 (邻接链)
    for i in range(nqubits - 1):
        c.add(gates.CZ(i, i + 1))
    # Mixer: X 旋转
    for i in range(nqubits):
        c.add(gates.RX(i, beta))
    return c


def build_supremacy(nqubits, depth=4):
    """
    构造类 Google Supremacy 随机线路:
    每层对每个 qubit 上单比特旋转 + 邻接 CZ 纠缠门交替
    depth 控制层数 (越深越难收缩)
    """
    import numpy as np
    rng = np.random.RandomState(2026)
    c = Circuit(nqubits)

    for layer in range(depth):
        # 单比特门: 随机 T 或 X^{1/2} 旋转
        for i in range(nqubits):
            if rng.random() > 0.5:
                c.add(gates.T(i))
            else:
                c.add(gates.RX(i, rng.uniform(0, 3.14)))

        # 纠缠门: CZ 按行/列交替 (模拟 2D 网格, 但 1D 链也够测试)
        # 偶数层: (0,1) (2,3) ...
        # 奇数层: (1,2) (3,4) ...
        if layer % 2 == 0:
            pairs = [(i, i + 1) for i in range(0, nqubits - 1, 2)]
        else:
            pairs = [(i, i + 1) for i in range(1, nqubits - 1, 2)]
        for a, b in pairs:
            c.add(gates.CZ(a, b))

    return c


def init_backend():
    """初始化 qibotn 的 qutensornet 后端 (纯 CPU)"""
    computation_settings = {
        "MPI_enabled": False,
        "MPS_enabled": False,
        "NCCL_enabled": False,
        "expectation_enabled": False,
    }
    qibo.set_backend(backend="qibotn", platform="qutensornet",
                     runcard=computation_settings)
    print(f"[INFO] 后端: qibotn/qutensornet (纯CPU)")


def run_single(workload, nqubits, depth=4):
    """构造并运行单个线路, 返回 (runtime, status, ngates)"""
    builder = {
        "QFT": lambda n: build_qft(n),
        "QAOA": lambda n: build_qaoa(n),
        "Supremacy": lambda n: build_supremacy(n, depth),
    }[workload]

    print(f"  构造 {workload}({nqubits}) ...", end="", flush=True)
    c = builder(nqubits)
    ngates = c.ngates
    print(f" {ngates} 门. 开始收缩...", end="", flush=True)

    t0 = time.perf_counter()
    try:
        result = c()
        runtime = time.perf_counter() - t0
        status = "success"
        # 简单验证: state vector 形状对不对
        sv = result.state()
        expected = 2 ** nqubits
        if sv.shape[0] != expected:
            status = f"FAIL:bad_shape_{sv.shape[0]}_expected_{expected}"
        print(f" 完成: {runtime:.2f}s [{status}]")
    except Exception as e:
        runtime = time.perf_counter() - t0
        status = f"FAIL:{type(e).__name__}"
        print(f" 失败: {runtime:.2f}s [{status}: {e}]")

    return runtime, status, ngates


def main():
    parser = argparse.ArgumentParser(description="QiboTN benchmark")
    parser.add_argument("--workloads", default="QFT,QAOA,Supremacy",
                        help="逗号分隔的 workload 列表")
    parser.add_argument("--qubits", default="8,10,12",
                        help="逗号分隔的 qubit 数列表")
    parser.add_argument("--depth", type=int, default=4,
                        help="Supremacy 线路深度 (默认4)")
    parser.add_argument("--tag", default="baseline",
                        help="本次运行的标签 (写入 csv)")
    parser.add_argument("--output", default="results.csv",
                        help="输出 CSV 文件路径")
    args = parser.parse_args()

    workloads = [w.strip() for w in args.workloads.split(",")]
    qubit_list = [int(q.strip()) for q in args.qubits.split(",")]

    print(f"=== QiboTN Benchmark ===")
    print(f"workloads: {workloads}")
    print(f"qubits: {qubit_list}")
    print(f"supremacy depth: {args.depth}")
    print(f"tag: {args.tag}")
    print(f"threads: OMP={os.environ.get('OMP_NUM_THREADS','未设')} "
          f"MKL={os.environ.get('MKL_NUM_THREADS','未设')}")
    print()

    init_backend()

    # 准备命令字符串 (记录到 csv 便于复现)
    threads_info = f"OMP={os.environ.get('OMP_NUM_THREADS','default')}"
    cmd_template = f"python benchmark.py --workloads {{w}} --qubits {{q}} --tag {args.tag} [{threads_info}]"

    rows = []
    # 从小到大跑 (先小后大, 让 numba JIT 编译只发生一次)
    for nq in qubit_list:
        for wl in workloads:
            cmd = cmd_template.format(w=wl, q=nq)
            runtime, status, ngates = run_single(wl, nq, args.depth)
            rows.append({
                "workload": wl,
                "qubits": nq,
                "ngates": ngates,
                "command": cmd,
                "runtime": f"{runtime:.4f}",
                "status": status,
                "tag": args.tag,
            })
            # 释放内存
            gc.collect()

    # 写 CSV
    write_header = not os.path.exists(args.output)
    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tag", "workload", "qubits", "ngates",
            "command", "runtime", "status"
        ])
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"\n=== 完成, 结果写入 {args.output} ===")
    print(f"{'workload':<12} {'qubits':<8} {'ngates':<8} {'runtime':<12} {'status'}")
    print("-" * 60)
    for r in rows:
        print(f"{r['workload']:<12} {r['qubits']:<8} {r['ngates']:<8} "
              f"{r['runtime']:<12} {r['status']}")


if __name__ == "__main__":
    main()
