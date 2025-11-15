# Batch Composition Optimization Framework

**统一的批组合优化框架，支持多种tile-based加速器（NPU/GPU/TPU）**

[![Status](https://img.shields.io/badge/Status-Core_Complete-green)]()
[![Python](https://img.shields.io/badge/Python-3.x-blue)]()
[![MLIR](https://img.shields.io/badge/MLIR-17.0+-red)]()

---

## 🚀 快速开始

```bash
# 1. 阅读核心文档
cat docs/summaries/IMPLEMENTATION_COMPLETE.md  # ⭐ 项目总览

# 2. 安装Python依赖
pip install -r requirements.txt

# 3. 测试Python优化器
cd python && python3 test_new_structure.py

# 4. 查看使用指南
cat docs/guides/QUICKSTART.md
```

---

## 📁 目录结构

```
npu_pass_implementation/
├── README.md                    # 本文件 - 项目总览
├── REFACTORING_PLAN.md         # 重构总计划
│
├── docs/                        # 📚 文档
│   ├── summaries/               # 各阶段总结
│   │   ├── IMPLEMENTATION_COMPLETE.md    # ⭐ 完整实施总结
│   │   ├── PHASE1_SUMMARY.md             # Phase 1: Python重构
│   │   ├── PHASE2_SUMMARY.md             # Phase 2: Dialect设计
│   │   ├── PHASE3_PLAN.md                # Phase 3: C++ Pass计划
│   │   └── PHASE3_PROGRESS.md            # Phase 3: 进度报告
│   │
│   ├── design/                  # 设计文档
│   │   └── BATCHCOMP_DIALECT_DESIGN.md   # ⭐ Dialect完整设计
│   │
│   ├── guides/                  # 使用指南
│   │   ├── QUICKSTART.md                 # 快速开始
│   │   ├── CP_SAT_README.md              # CP-SAT算法说明
│   │   └── CP_SAT_INTEGRATION_GUIDE.md   # CP-SAT集成指南
│   ├── tutorials/               # 教程文档
│   ├── api/                     # API参考
│   └── development/             # 开发指南
│
├── include/                     # C++ 头文件
│   ├── BatchComp/               # ⭐ BatchComp Dialect（新）
│   │   ├── BatchCompOps.td      # 操作定义（600行）
│   │   ├── BatchCompPasses.td   # Pass定义（200行）
│   │   └── BatchCompPasses.h    # Pass声明
│   └── NPU/                     # NPU Dialect（旧，保留）
│       └── ...
│
├── lib/                         # C++ 实现
│   ├── BatchComp/               # ⭐ BatchComp Pass实现（新）
│   │   ├── BatchCompTileScheduler.cpp    # 主Pass（400行）
│   │   └── PythonIntegration.cpp         # Python集成（530行）
│   └── NPU/                     # NPU Pass实现（旧）
│       └── ...
│
├── python/                      # ⭐ Python优化器（新架构）
│   ├── core/                    # 核心模块
│   │   ├── tile_info.py         # Tile数据结构
│   │   ├── hardware_config.py   # 硬件配置系统
│   │   └── batch_optimizer.py   # 优化器基类
│   ├── optimizers/              # 优化算法
│   │   ├── greedy.py            # 贪心算法
│   │   ├── improved_greedy.py   # 改进贪心
│   │   └── cpsat_optimizer.py   # CP-SAT优化器
│   └── interface/               # C++接口
│       └── cpp_interface.py     # Python→C++接口
│
├── configs/                     # 硬件配置文件
│   ├── npu.yaml                 # NPU配置
│   ├── gpu.yaml                 # GPU配置
│   └── tpu.yaml                 # TPU配置
│
└── requirements.txt             # Python依赖
```

---

## 📖 核心文档（必读）

### 1. 项目总览 ⭐
- **[IMPLEMENTATION_COMPLETE.md](docs/summaries/IMPLEMENTATION_COMPLETE.md)**
  - 完整的实施总结
  - 包含所有Phases的概览
  - 架构图和性能数据
  - **推荐首先阅读**

### 2. Dialect设计 ⭐
- **[BATCHCOMP_DIALECT_DESIGN.md](docs/design/BATCHCOMP_DIALECT_DESIGN.md)**
  - BatchComp Dialect完整设计
  - 15个操作的详细说明
  - MLIR转换示例
  - 硬件参数化方案

### 3. 阶段总结
- [PHASE1_SUMMARY.md](docs/summaries/PHASE1_SUMMARY.md) - Python优化器重构
- [PHASE2_SUMMARY.md](docs/summaries/PHASE2_SUMMARY.md) - Dialect设计
- [PHASE3_PLAN.md](docs/summaries/PHASE3_PLAN.md) - C++ Pass实施计划

---

## 🎯 关键特性

### 1. 多硬件支持

| 硬件 | Batch Size | Must Fill | 特性 |
|------|-----------|-----------|------|
| **NPU** | 16 | Yes | 16-way parallel |
| **GPU** | 32 | No | Warp-based (Tensor Core) |
| **TPU** | 128 | No | 128×128 systolic array |

### 2. 三种优化算法

| 算法 | 复杂度 | 适用场景 | 性能 |
|------|--------|----------|------|
| **Greedy** | O(n) | 单矩阵、快速编译 | 基准 |
| **ImprovedGreedy** | O(n log n) | 多矩阵、中小规模 | Batch-20%, NOP-40% |
| **CP-SAT** | NP (有时限) | 多矩阵、中等规模 | **Batch-40%, NOP-80%** |

### 3. 两级IR抽象

**高层（推荐）** - 简洁易用：
```mlir
%tileset = batch_comp.generate_tiles %A, %B {tile_size = [16,16,16]}
%schedule = batch_comp.schedule_batches %tileset {hardware = "npu"}
%C = batch_comp.execute_schedule %schedule
```

**底层（完全控制）** - 精细调优：
```mlir
%tile = batch_comp.slice %A[0, 0] [16, 16]
%batch = batch_comp.create_batch {batch_size = 16}
%result = batch_comp.execute_batch %batch
```

---

## 💻 使用示例

### Python API

```python
from python.core import HardwareConfig, UnifiedOptimizer

# 加载配置
config = HardwareConfig.from_yaml("configs/npu.yaml")

# 创建优化器
optimizer = UnifiedOptimizer(config)

# 优化tiles
result = optimizer.optimize(tiles, algorithm="cpsat")
print(f"Batches: {result.num_batches}, Utilization: {result.utilization:.2%}")
```

### MLIR Pass

```bash
# NPU优化
mlir-opt input.mlir \
  --batch-comp-tile-scheduler="hardware=npu tile-size=16,16,16 algorithm=cpsat"

# GPU优化
mlir-opt input.mlir \
  --batch-comp-tile-scheduler="hardware=gpu tile-size=16,16,16"
```

---

## 📊 性能数据

| 场景 | Tiles | 算法 | Batches | 利用率 | 改进 |
|------|-------|------|---------|--------|------|
| 单矩阵 | 20 | Greedy | 2 | 62.5% | 基准 |
| 多矩阵 | 45 | ImprovedGreedy | 3 | 93.75% | +31% |
| vs简单贪心 | 45 | CP-SAT | 3 vs 5 | 93.75% vs 75% | **Batch -40%<br/>NOP -80%** |

---

## 🔧 技术栈

| 组件 | 技术 | 代码量 |
|------|------|--------|
| **MLIR Dialect** | TableGen | 600行 |
| **C++ Pass** | MLIR Pass Infrastructure | 1200行 |
| **Python优化器** | Python 3.x | 2000行 |
| **约束求解** | Google OR-Tools CP-SAT | - |
| **Python-C++集成** | pybind11 | 530行 |
| **配置管理** | YAML | - |

---

## 📈 开发状态

| 组件 | 状态 | 完成度 |
|------|------|--------|
| **Python优化器** | ✅ 完成 | 100% |
| **Dialect定义** | ✅ 完成 | 100% |
| **Pass框架** | ✅ 完成 | 100% |
| **Python-C++集成** | ✅ 完成 | 100% |
| **IR生成** | ✅ 完成 | 100% |
| **Lowering Pass** | ✅ 框架完成 | 80% |
| **NPU Backend Pass** | ✅ 框架完成 | 80% |
| **Canonicalize Pass** | ✅ 完成 | 100% |
| **Dialect实现文件** | ⏳ 待创建 | 0% |
| **端到端测试** | ⏳ 待实现 | 0% |

**总体完成度**: 85%

---

## 🛠️ 下一步工作

### 高优先级（剩余15%）
1. ✅ Python优化器重构
2. ✅ Dialect设计
3. ✅ Pass框架实现
4. ✅ Python-C++集成
5. ✅ IR生成逻辑完成
6. ✅ BatchCompLoweringPass框架
7. ✅ BatchCompToNPUPass框架
8. ✅ BatchCompCanonicalizePass
9. ✅ CMakeLists.txt配置
10. ⏳ 创建Dialect实现文件（BatchCompDialect.cpp, BatchCompOps.cpp, BatchCompTypes.cpp）
11. ⏳ 完善Lowering Pass的具体实现
12. ⏳ 端到端测试

### 中优先级
- 集成测试与性能benchmarking
- API文档生成
- 使用教程编写

### 低优先级
- GPU/TPU后端完整实现
- 运行时库集成
- 高级优化（缓存、并行）

---

## 📚 参考资料

### 内部文档
- [重构计划](REFACTORING_PLAN.md)
- [完整实施总结](docs/summaries/IMPLEMENTATION_COMPLETE.md)
- [Dialect设计](docs/design/BATCHCOMP_DIALECT_DESIGN.md)

### 外部资源
- [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/) - CP-SAT算法参考
- [MLIR Documentation](https://mlir.llvm.org/) - MLIR官方文档
- [Google OR-Tools](https://developers.google.com/optimization) - OR-Tools文档

---

## 👥 贡献者

NPU Compiler Team

---

## 📄 许可

[添加许可信息]

---

**最后更新**: 2025-11-15
**项目状态**: 核心实现完成，进入测试阶段
**下一个里程碑**: 端到端测试完成
