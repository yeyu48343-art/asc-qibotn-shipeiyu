#!/bin/bash
# 服务器版规模套件: 100GB 内存, dense 全曲线 + MPS + 线程对照 + fidelity
# 每项结果独立落盘到 /root/results/results_scale.csv, 断点可续跑
cd /root/benchmarks
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=/root/venv/bin/python

mkdir -p /root/results
[ -f /root/results/results_scale.csv ] || cp results_scale.seed.csv /root/results/results_scale.csv

S=/root/results/results_scale.csv
done_tags() { [ -f "$S" ] && cut -d, -f1 "$S" | tail -n +2 | sort -u; }
todo() { done_tags | grep -qx "$1" && { echo "[skip] $1"; return 1; }; return 0; }

echo "=== server suite start $(date +%H:%M:%S) ==="

# A. dense 深度曲线 (100GB 内存, 超时给足)
todo A_dense_d50  && $PY run_scale_suite.py --single A_dense_d50  Supremacy 28 50  dense 0 3600 4
todo A_dense_d100 && $PY run_scale_suite.py --single A_dense_d100 Supremacy 28 100 dense 0 5400 4

# A2. QFT 30q dense (16GB 态向量, 服务器上应可完成)
todo A2_qft30_dense && $PY run_scale_suite.py --single A2_qft30_dense QFT 30 0 dense 0 3600 4

# B. MPS 对照 (8 项)
for d in 8 30 50 100; do
  for b in 128 256; do
    todo B_mps_d${d}_b${b} && $PY run_scale_suite.py --single B_mps_d${d}_b${b} Supremacy 28 $d mps $b 1200 4
  done
done

# C. QFT 30q MPS
todo C_qft30_mps256 && $PY run_scale_suite.py --single C_qft30_mps256 QFT 30 0 mps 256 1200 4

# D. 线程对照 (d8 dense, t4 已有 A_dense_d8)
todo D_dense_t12 && $PY run_scale_suite.py --single D_dense_t12 Supremacy 28 8 dense 0 900 12
todo D_dense_t20 && $PY run_scale_suite.py --single D_dense_t20 Supremacy 28 8 dense 0 900 20

# F. fidelity 量化 (QFT 24q)
todo F_fidelity && $PY benchmark_fidelity.py && touch /root/results/.fidelity_done

echo "=== server suite done $(date +%H:%M:%S) ==="
