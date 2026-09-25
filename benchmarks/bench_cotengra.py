#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
收缩路径优化实验 (修正版): 直接把 optimize 传给 TensorNetwork.contract()
对比 greedy(快速贪心) vs auto-hq(cotengra 超优化) 的路径质量差异

理论依据: Gray & Kourtis, "Hyper-optimized tensor network contraction",
Quantum 5, 410 (2021) — 收缩顺序决定总计算量, 超优化搜索可显著降低 cost。
"""
import csv
import os
import sys
import time

import numpy as np
import quimb.tensor as qtn


def build_qft(n):
    circ = qtn.Circuit(n)
    for i in range(n):
        circ.h(i)
        for j in range(i + 1, n):
            circ.cu1(np.pi / (2 ** (j - i)), i, j)
    for i in range(n // 2):
        circ.swap(i, n - 1 - i)
    return circ


def build_supremacy(n, depth):
    rng = np.random.RandomState(2026)
    circ = qtn.Circuit(n)
    for layer in range(depth):
        for i in range(n):
            if rng.random() > 0.5:
                circ.rz(i, np.pi / 4)
            else:
                circ.rx(i, rng.uniform(0, 3.14))
        pairs = (([(i, i + 1) for i in range(0, n - 1, 2)]) if layer % 2 == 0
                 else [(i, i + 1) for i in range(1, n - 1, 2)])
        for a, b in pairs:
            circ.cz(a, b)
    return circ


def timed_contract(circ, optimize):
    psi = circ.psi.full_simplify(seq="DRC")
    t0 = time.perf_counter()
    result = psi.contract(all, optimize=optimize)
    runtime = time.perf_counter() - t0
    return runtime, float(np.abs(result.data).max())


def main():
    out_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "results", "results_cotengra.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    jobs = [
        ("QFT", 28, 0),
        ("Supremacy", 28, 8),
    ]
    for wl, n, d in jobs:
        for opt in ["greedy", "auto-hq"]:
            circ = build_qft(n) if wl == "QFT" else build_supremacy(n, d)
            ngates = len(circ.gates)
            print(f"[run] {wl}({n},d{d}) optimize={opt} gates={ngates} ...",
                  flush=True)
            t0 = time.perf_counter()
            try:
                rt, peak = timed_contract(circ, opt)
                status = "success"
                print(f"      total={time.perf_counter()-t0:.2f}s "
                      f"contract={rt:.2f}s max_amp={peak:.3f}", flush=True)
            except Exception as e:
                rt = time.perf_counter() - t0
                status = f"FAIL:{type(e).__name__}"
                print(f"      fail {rt:.2f}s {status}: {str(e)[:80]}", flush=True)
            row = {"tag": f"cotengra_{opt.replace('-', '_')}", "workload": wl,
                   "qubits": n, "depth": d, "mode": "dense",
                   "bond": 0, "threads": os.environ.get("OMP_NUM_THREADS", "?"),
                   "ngates": ngates, "runtime": f"{rt:.4f}", "status": status,
                   "detail": f"optimize={opt}"}
            write_header = not os.path.exists(out_csv)
            with open(out_csv, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=[
                    "tag", "workload", "qubits", "depth", "mode", "bond",
                    "threads", "ngates", "runtime", "status", "detail"])
                if write_header:
                    w.writeheader()
                w.writerow(row)


if __name__ == "__main__":
    main()
