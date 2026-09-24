#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第二批服务器实验: 大内存下的 dense 补全 + 同机 t4 基准 + Supremacy fidelity"""
import subprocess
import sys
import time

PY = "/root/venv/bin/python"
HERE = "/root/benchmarks"


def run_one(tag, workload, nqubits, depth, mode, bond, timeout, threads=4):
    import csv
    import os
    import json
    env = {"OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads),
           "PATH": "/usr/bin:/bin:/root/venv/bin"}
    cmd = [PY, f"{HERE}/benchmark_scale.py", workload, str(nqubits),
           str(depth), mode, str(bond)]
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
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            p.kill()
        row = {"workload": workload, "qubits": nqubits, "depth": depth,
               "mode": mode, "bond": bond, "ngates": "",
               "runtime": timeout, "status": "TIMEOUT",
               "detail": f"exceeded {timeout}s"}
    row["tag"] = tag
    row["threads"] = threads
    out_csv = "/root/results/results_scale.csv"
    fields = ["tag", "workload", "qubits", "depth", "mode", "bond",
              "threads", "ngates", "runtime", "status", "detail"]
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[{tag}] {workload}({nqubits},d{depth},{mode}) t{threads}: "
          f"{row['runtime']}s -> {row['status']}", flush=True)


def main():
    print("=== server batch 2 start ===", flush=True)
    # 同机 t4 基准 (与 t12/t20 对照可比)
    run_one("D_dense_t4_srv", "Supremacy", 28, 8, "dense", 0, 900, 4)
    # 大内存下的 dense 补全
    run_one("A_dense_d50_srv", "Supremacy", 28, 50, "dense", 0, 3600, 4)
    run_one("A_dense_d100_srv", "Supremacy", 28, 100, "dense", 0, 5400, 4)
    print("=== batch2 dense done, start fidelity ===", flush=True)
    # Supremacy fidelity (与 QFT 的 fidelity=1.0 对照)
    for bond in [128, 256]:
        p = subprocess.run([PY, f"{HERE}/benchmark_fidelity2.py",
                            "Supremacy", "24", "8", str(bond)],
                           capture_output=True, text=True, timeout=1200)
        print(p.stdout.strip(), flush=True)
    print("=== server batch 2 done ===", flush=True)


if __name__ == "__main__":
    main()
