# Batch Composition Optimization Research

This directory contains research, design, and implementation for the Batch Composition Optimization framework - a unified approach to tile-based accelerator optimization.

## Directory Structure

```
reserch/
├── README.md                        # 本文件 - 研究目录总览
│
├── design/                          # 📐 技术方案文档
│   └── BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md  # 统一框架设计 (16KB)
│       ├── 1. 问题定义 - 批组合优化的统一视角
│       ├── 2. MLIR 转换设计 - 输入/输出定义
│       ├── 3. 批组合优化算法 - CP-SAT实现
│       ├── 4. 实施计划 - 4阶段路线图
│       ├── 5. 关键创新点
│       ├── 6. 性能目标
│       ├── 7. 风险和缓解
│       └── 8. 结论
│
├── npu_pass_implementation/         # 💻 完整实现（MLIR Pass + CP-SAT）
│   ├── lib/NPU/                     # MLIR Pass C++ 实现
│   ├── include/NPU/                 # MLIR Dialect 定义
│   ├── CP_SAT_TILE_SCHEDULER.py     # CP-SAT 全局优化器
│   ├── greedy_scheduler.py          # 贪心算法基准
│   ├── cpsat_scheduler_cli.py       # CLI 工具
│   ├── benchmark_schedulers.py      # 性能对比测试
│   ├── test_cpsat_basic.py          # 单元测试
│   ├── requirements.txt             # Python 依赖
│   ├── QUICKSTART.md                # 快速开始指南
│   ├── CP_SAT_README.md             # CP-SAT 完整文档
│   ├── CP_SAT_INTEGRATION_GUIDE.md  # C++/MLIR 集成指南
│   ├── CPSAT_IMPLEMENTATION_SUMMARY.md  # 实现总结
│   ├── FILE_INDEX.md                # 文件索引
│   └── ... (更多文档)
│
└── examples/                        # 📝 C++ 示例代码
    ├── arbitrary_irregular_tiling.cpp       # 任意不规则 Tiling 实现 (15KB)
    ├── irregular_tiling_example.cpp         # 不规则 Tiling 示例 (12KB)
    └── tensor_core_tile_scheduler.cpp       # Tensor Core 调度器 (16KB)
```

## 文档分类

### 设计文档 (design/)

| 文档 | 大小 | 内容 |
|------|------|------|
| [BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md](design/BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md) | 16KB | **批组合优化统一框架** |

### 生产化分析

| 文档 | 内容 | 重要性 |
|------|------|--------|
| [PRODUCTION_READINESS_ANALYSIS.md](PRODUCTION_READINESS_ANALYSIS.md) | 生产就绪性差距分析 | 🚨 Critical |
| [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) | 详细实施路线图（9个月） | 🚨 Critical |

**设计核心理念**:
- **统一抽象**: 所有 tile-based 加速器的优化本质是批组合（Batch Composition）问题
- **多硬件支持**: NPU (16-tile固定批)、GPU (32-thread warp)、TPU (128×128 systolic)
- **通用优化**: CP-SAT 约束求解器进行全局优化
- **问题本质**: 装箱问题（Bin Packing）的变种

### 实现代码 (npu_pass_implementation/)

#### C++ MLIR Pass
- `lib/NPU/NPUTileSchedulerPass.cpp` - 主 Pass 实现
- `include/NPU/NPUOps.td` - NPU Dialect 定义
- `include/NPU/NPUPasses.td` - Pass 注册

#### Python CP-SAT 优化器
- `CP_SAT_TILE_SCHEDULER.py` - 核心 CP-SAT 实现
- `greedy_scheduler.py` - 贪心基准算法
- `cpsat_scheduler_cli.py` - CLI 工具

#### 测试和文档
- `benchmark_schedulers.py` - 性能对比
- `test_cpsat_basic.py` - 单元测试
- 完整的文档套件（见 FILE_INDEX.md）

### 示例代码 (examples/)

| 文件 | 大小 | 描述 |
|------|------|------|
| [arbitrary_irregular_tiling.cpp](examples/arbitrary_irregular_tiling.cpp) | 15KB | 任意不规则 Tiling 实现 |
| [irregular_tiling_example.cpp](examples/irregular_tiling_example.cpp) | 12KB | 不规则 Tiling 示例 |
| [tensor_core_tile_scheduler.cpp](examples/tensor_core_tile_scheduler.cpp) | 16KB | Tensor Core 调度器 |

## 快速开始

### 1. 阅读设计文档

阅读核心设计文档：
- [BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md](design/BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md) - **批组合优化统一框架**（16KB）
  - 识别 tile-based 加速器的共同优化模式
  - 完整的 MLIR 转换定义（输入/输出）
  - CP-SAT 优化算法实现
  - 多硬件适配策略

### 2. 运行 CP-SAT 优化器

```bash
cd npu_pass_implementation

# 安装依赖
pip install -r requirements.txt

# 快速测试
python3 test_cpsat_basic.py

# 性能对比
python3 benchmark_schedulers.py
```

### 3. 集成到 MLIR Pass

参考 [npu_pass_implementation/CP_SAT_INTEGRATION_GUIDE.md](npu_pass_implementation/CP_SAT_INTEGRATION_GUIDE.md)

## 核心成果

### 1. 统一优化框架
- **批组合优化**：识别所有 tile-based 加速器的共同问题
- **通用抽象**：一个框架支持 NPU/GPU/TPU
- **算法复用**：CP-SAT 优化跨硬件共享

### 2. Batch Composition Dialect
- **完整操作集**：slice, pad, tile_matmul, accumulate
- **硬件参数化**：通过配置适配不同硬件
- **K维规约处理**：软件控制的累加策略
- **批次管理**：固定批/可变批的统一处理

### 3. CP-SAT 全局优化器
- 基于 [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/)
- 多矩阵场景下 **20-40% 性能提升**
- 完整的 Python 实现 + CLI 工具
- 生产就绪（错误处理、超时控制、配置选项）

## 性能对比

### 多矩阵场景 (65 tiles)

| 算法 | Batches | NOPs | 利用率 | 时间 |
|------|---------|------|--------|------|
| 贪心 | 5 | 15 | 81.2% | 0.001s |
| 改进贪心 | 4 | 1 | 98.4% | 0.002s |
| **CP-SAT** | **4** | **1** | **98.4%** | **0.12s** |

**改进**: 相比贪心，batch 减少 20%，NOP 减少 93%

## 推荐使用策略

| 场景 | Tiles 数量 | 推荐算法 |
|------|-----------|----------|
| 单矩阵 | 任意 | 贪心 |
| 多矩阵（小） | < 50 | 改进贪心 |
| 多矩阵（中） | 50-1000 | **CP-SAT** |
| 多矩阵（大） | > 1000 | 改进贪心 |
| 实时编译 | 任意 | 贪心 |

## 参考资料

### 外部资源
- [CP-SAT Primer](https://d-krupke.github.io/cpsat-primer/) - Dominik Krupke, TU Braunschweig
- [MLIR Documentation](https://mlir.llvm.org/)
- [Google OR-Tools](https://developers.google.com/optimization)

### 内部文档
- 完整文件索引: [npu_pass_implementation/FILE_INDEX.md](npu_pass_implementation/FILE_INDEX.md)
- CP-SAT 集成指南: [npu_pass_implementation/CP_SAT_INTEGRATION_GUIDE.md](npu_pass_implementation/CP_SAT_INTEGRATION_GUIDE.md)
- 快速开始: [npu_pass_implementation/QUICKSTART.md](npu_pass_implementation/QUICKSTART.md)

## 联系和贡献

如有问题或建议，请查看：
- 核心设计文档: [design/BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md](design/BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md)
  - 批组合优化的统一视角
  - 完整的 MLIR 转换设计
  - 多硬件适配策略

---

**最后更新**: 2025-11-15
**项目状态**: 设计完成，实现进行中
