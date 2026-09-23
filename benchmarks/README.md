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
├── src/                ← 官方源码
├── examples/           ← 官方示例
├── tests/              ← 官方测试
├── benchmarks/         ← ★ 新增：本作业的实验脚本
│   ├── README.md       ← 本文件
│   ├── benchmark.py    ← 主 benchmark（QFT/Supremacy 构造 + 计时 + results.csv）
│   └── benchmark_mps.py← MPS 近似实验（qibotn 原生 mps_opts 路径）
└── results/            ← ★ 新增：实验结果
    └── results.csv     ← 全部计时数据（含 baseline 与各优化）
```

## 运行环境

| 项目 | 配置 |
|------|------|
| 系统 | Windows 11 + WSL2 Ubuntu 24.04 |
| CPU | Intel i5-12500H（4P+8E，16 逻辑核） |
| 可用内存 | 7.6 GB（WSL2 默认分配） |
| Python | 3.11.15（conda 独立环境） |
| qibotn | 0.0.3（PyPI 纯 CPU 版；官方 main 分支 pyproject 误将 cuda-toolkit 列为必需，故未用 poetry 装源码依赖，实验经 PyPI 发行版运行，脚本与版本在 results.csv command 列可复现） |
| 后端 | qibotn/qutensornet（纯 CPU，赛题规定路径，未使用 qibojit/cuQuantum/GPU） |

## 实验设计（对齐评审要求）

1. **规模**：从 20 qubits 逐级加大到 28/30 qubits；Supremacy 线路 depth 从 8 加深至 30/50/100（gate 数从数百提升至数千量级），直至触及本机内存/时间上限
2. **优化对照**：在相同 workload、相同精度、相同计算路径下仅改单一变量（线程数 / BLAS 环境变量）；MPS 作为"可扩展性方案"单独归档，并以 fidelity 量化其精度代价，不与精确收缩的加速比混算
3. **正确性**：state vector 形状校验 + MPS 与精确态的 fidelity 对比
4. **诚实记录**：超时/未完成/失败的运行同样写入 results（status 列），作为规模压力的证据

详见 results/results.csv 与最终报告 PDF。
