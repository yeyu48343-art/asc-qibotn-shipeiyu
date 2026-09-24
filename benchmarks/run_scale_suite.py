#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
规模升级实验套件驱动器 (对齐评审要求: 30 qubits / 数千 gates / 数百层 depth)

每项实验以独立子进程运行 (超时控制 + 内存隔离), 结果追加写入 results/results_scale.csv

实验矩阵:
  A. Deep Supremacy 28q, depth 8/30/50/100, dense 精确收缩 (含超时)
  B. 同 workload, MPS bond=128/256 (可扩展性方案)
  C. QFT 30q: dense (预期 OOM, 可扩展性上限证据) vs MPS bond=256
  D. 线程对照: 28q d8 Supremacy dense, OMP=4 vs OMP=12 (合规个人优化)
"""
import csv
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "benchmark_scale.py")
OUT = os.path.join(HERE, "..", "results", "results_scale.csv")

BASE_ENV = {"OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}


def run_one(tag, workload, nqubits, depth, mode, bond, timeout, threads=4):
    env = os.environ.copy()
    env.update(BASE_ENV)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)

    cmd = [sys.executable, SCRIPT, workload, str(nqubits), str(depth), mode, str(bond)]
    t0 = time.time()

    import signal
    p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=HERE,
                         start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
        line = out.strip().splitlines()[-1] if out.strip() else ""
        row = json.loads(line) if line.startswith("{") else {
            "workload": workload, "qubits": nqubits, "depth": depth,
            "mode": mode, "bond": bond, "ngates": "",
            "runtime": round(time.time() - t0, 1), "status": "FAIL:crash",
            "detail": (err or "")[:100]}
    except subprocess.TimeoutExpired:
        # 先强杀整个进程组 (避免 quimb 子进程挂死拖住清理)
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            p.kill()
        try:
            p.communicate(timeout=30)
        except Exception:
            pass
        row = {"workload": workload, "qubits": nqubits, "depth": depth,
               "mode": mode, "bond": bond, "ngates": "",
               "runtime": timeout, "status": "TIMEOUT",
               "detail": f"exceeded {timeout}s"}
    row["tag"] = tag
    row["threads"] = threads

    fields = ["tag", "workload", "qubits", "depth", "mode", "bond",
              "threads", "ngates", "runtime", "status", "detail"]
    write_header = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{tag}] {workload}({nqubits},d{depth},{mode}"
          + (f",bond{bond}" if mode == "mps" else "")
          + f",t{threads}): {row['runtime']}s -> {row['status']}", flush=True)
    return row


def done_tags():
    tags = set()
    if os.path.exists(OUT):
        with open(OUT, newline="") as f:
            for row in csv.DictReader(f):
                tags.add(row.get("tag"))
    return tags


def main():
    # --single 模式: 每次只跑一个实验 (前台短跑, 适配不稳定的会话)
    if len(sys.argv) >= 8 and sys.argv[1] == "--single":
        tag, workload, nq, depth, mode, bond, timeout = sys.argv[2:9]
        threads = int(sys.argv[9]) if len(sys.argv) > 9 else 4
        if tag in done_tags():
            print(f"[skip] {tag} (already in csv)", flush=True)
            return
        run_one(tag, workload, int(nq), int(depth), mode, int(bond),
                timeout=int(timeout), threads=threads)
        return

    print("=== 规模升级实验套件 (评审对齐版, 断点续跑) ===", flush=True)
    print(f"start: {time.strftime('%H:%M:%S')}", flush=True)
    done = done_tags()

    def todo(tag):
        if tag in done:
            print(f"[skip] {tag} (already in csv)", flush=True)
            return False
        return True

    # A. Deep Supremacy dense 精确收缩 (超时上限 25 分钟)
    for d in [8, 30, 50, 100]:
        if todo(f"A_dense_d{d}"):
            run_one(f"A_dense_d{d}", "Supremacy", 28, d, "dense", 0, timeout=1500)

    # B. 同 workload MPS (超时 15 分钟)
    for d in [8, 30, 50, 100]:
        for bond in [128, 256]:
            if todo(f"B_mps_d{d}_b{bond}"):
                run_one(f"B_mps_d{d}_b{bond}", "Supremacy", 28, d, "mps", bond,
                        timeout=900)

    # C. QFT 30q: dense 预期 OOM / MPS 应可运行
    if todo("C_qft30_dense"):
        run_one("C_qft30_dense", "QFT", 30, 0, "dense", 0, timeout=600)
    if todo("C_qft30_mps256"):
        run_one("C_qft30_mps256", "QFT", 30, 0, "mps", 256, timeout=900)

    # D. 线程对照 (合规个人优化): 28q d8 Supremacy dense
    if todo("D_dense_t4"):
        run_one("D_dense_t4", "Supremacy", 28, 8, "dense", 0, timeout=1500, threads=4)
    if todo("D_dense_t12"):
        run_one("D_dense_t12", "Supremacy", 28, 8, "dense", 0, timeout=1500, threads=12)

    print(f"done: {time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
