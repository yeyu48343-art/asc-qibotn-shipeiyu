# QiboTN 选拔作业实验（基于官方源码仓库）

**姓名**：石培玉　**学号**：240809010102　**年级专业**：计算机科学与技术 24 级

## 源码出处（Provenance）

本仓库基于官方赛题源码仓库开展：

- 官方仓库：<https://github.com/qiboteam/qibotn>
- 基线 commit：`7ebbd4d4a00f37b898c552922953591b78b0af7e`（2026-09-22）
- 获取方式：GitHub API tarball（`/repos/qiboteam/qibotn/tarball/main`）
- 本仓库在官方源码之上**仅新增** `benchmarks/` 与 `results/` 两个目录，未修改官方 `src/` 任何代码

## 目录说明

```
/                       ← 官方 qibotn 源码（未改动）
├── src/  examples/  tests/  doc/
├── benchmarks/         ← ★ 新增：本作业的实验脚本
│   ├── README.md               ← 本文件
│   ├── benchmark.py            ← 主 benchmark（QFT/Supremacy 构造 + 计时 + CSV）
│   ├── benchmark_mps.py        ← MPS 近似实验（qibotn 原生 mps_opts 路径）
│   ├── benchmark_scale.py      ← 单次实验执行器（子进程调用，JSON 输出）
│   ├── run_scale_suite.py      ← 规模升级套件驱动（断点续跑 + --single 模式）
│   ├── benchmark_fidelity.py   ← fidelity 量化（QFT 24q）
│   ├── benchmark_fidelity2.py  ← fidelity 通用版（任意 workload/qubits/depth/bond）
│   └── server_suite.sh / server_batch2.py  ← 服务器端批量驱动
└── results/            ← ★ 新增：实验结果
    ├── results.csv             ← 笔记本端 QFT28 优化对照（baseline/线程/BLAS/MPS）
    ├── results_scale.csv       ← 规模升级全量数据（笔记本 + 服务器）
    ├── results_scale.server.csv← 服务器端副本
    ├── fidelity_*.json         ← fidelity 定量结果
    └── server_*.log            ← 服务器端原始运行日志
```

## 运行环境（两台，均 qibotn/qutensornet 纯 CPU 路径，未使用 GPU）

| 项目 | 机器 A（笔记本） | 机器 B（服务器） |
|------|----------------|----------------|
| 系统 | Windows 11 + WSL2 Ubuntu 24.04 | Debian 12（容器） |
| CPU | Intel i5-12500H（4P+8E，16 逻辑核） | 20 核（EPYC 级） |
| 可用内存 | 7.6 GB（WSL2 默认分配） | 100 GB |
| Python | 3.11.15（conda） | 3.12.3（venv） |
| qibotn | 0.0.3（PyPI 纯 CPU 版） | 0.0.3（PyPI 纯 CPU 版） |

> 注：官方 main 分支 pyproject 误将 cuda-toolkit 列为必需依赖，故经 PyPI 发行版安装运行；服务器配有 GPU 但实验全程未使用。

## 实验结果

### 1. 规模探索（笔记本，OMP=4）

| qubits | QFT(s) | Supremacy d8(s) | 评价 |
|--------|--------|-----------------|------|
| 20 | 0.37 | 0.62 | 太快 |
| 24 | 0.59 | 0.77 | 太快 |
| **28** | **97.71** | **80.04** | 用于优化对照 |

### 2. 规模升级与内存墙（Supremacy 28q，OMP=4）

| depth | gates | 层数 | 笔记本 7.6GB | 服务器 100GB | MPS b128（服务器） |
|-------|-------|------|-------------|-------------|--------------------|
| 8 | 332 | ~16 | 58.2s | 4.5s | 9.1s |
| 30 | 1245 | ~60 | 182.5s | — | 8.9s |
| 50 | 2075 | ~100 | **OOM ×3** | 46.4s | 8.7s |
| 100 | 4150 | ~200 | — | **崩溃 (1487.6s)** | **20.1s** |

- depth=50 精确收缩 3 次尝试均压垮 7.6GB WSL 虚拟机（OOM）——**内存墙实测**
- 同负载在 100GB 服务器 46.4s 完成，证实 OOM 纯由内存致因
- depth=100 精确收缩即使 100GB 也在 24.8 分钟后崩溃；**MPS 20.1s 完成**（越墙）
- QFT(30)（态向量 2^30 ≈ 10.7 亿维）服务器精确 dense 33.5s、MPS(b256) 26.5s——**30 qubits 达成**

### 3. fidelity 定量验证（回应"正确性"质疑）

| workload | 精确耗时 | MPS 耗时 | fidelity | 最大逐点误差 |
|----------|---------|---------|----------|--------------|
| QFT(24) | 0.86s | 0.85-0.86s | **1.00000000** | 4.8e-17 |
| Supremacy(24,d8) | 0.75-0.80s | 0.34s | **1.0000000000** | 6.8e-16 |

本规模下 MPS（bond=64/128/256）与精确态 fidelity 均为 1.0（机器精度），即 MPS 加速属于**同一数值精度下的合法对照结果**，未引入可测量精度损失。

### 4. 线程对照（合规个人优化）

| 环境 | workload | t4 | t12 | t20 |
|------|----------|----|----|-----|
| 笔记本（QFT28 dense） | 420 gates | 97.71s | 79.89s (1.22x) | — |
| 服务器（Supremacy d8 dense） | 332 gates | 4.52s | 5.25s | 4.37s |

线程收益依赖 workload 规模：笔记本大规模下 1.22x；服务器小规模上线程开销反超收益（t12 略慢于 t4）。

### 5. 收缩路径对比（Gray & Kourtis, Quantum 2021 方法）

同 workload、同精度、仅改变收缩顺序（`TensorNetwork.contract(optimize=...)`）：

| workload | greedy | auto-hq (cotengra) | auto (opt_einsum) | 结论 |
|----------|--------|--------------------|-------------------|------|
| QFT(28) | 22.41s | 11.67s | **8.67s** | 路径选择带来 **2.6×** 耗时差 |
| Supremacy(24,d30) | MemoryError | MemoryError | — | 高纠缠下精确收缩路径也无解 |

- 路径质量对性能影响巨大（2.6×），且**超优化不必然胜出**（QFT 规则结构下 opt_einsum 默认启发式已最优）——印证 Markov & Shi (SIAM J. Comput. 2008) 收缩复杂度理论
- 高纠缠深线路（24q/d30/1065 gates）连最优路径也无法在 7.6GB 内完成——**这正是 MPS 方法的适用区**（28q/d100 MPS 仅 20.1s），两种方法互补

## 复现方式

```bash
# 环境（两台机器相同步骤）
python3 -m venv ~/venv && ~/venv/bin/pip install qibotn

# 服务器端一键套件（断点续跑）
cd /root/benchmarks && nohup bash server_suite.sh > /root/suite.log 2>&1 &
# 第二批（大内存 dense 补全 + Supremacy fidelity）
nohup /root/venv/bin/python -u server_batch2.py > /root/batch2.log 2>&1 &

# fidelity 单项
/root/venv/bin/python benchmark_fidelity2.py QFT 24 0 256
```

## 注意事项

- **qibotn 版本**：必须用 PyPI 0.0.3。GitHub main 的 pyproject 误将 cuda-toolkit 列为必需依赖。
- **后端**：`platform="qutensornet"`（纯 CPU）。全程未使用 qibojit/cuQuantum/GPU。
- **诚实记录**：results_scale.csv 中 OOM / TIMEOUT / SKIPPED / FAIL 行均为真实观测，构成规模压力证据链。
