# 批组合优化框架重构计划

## 1. 目标架构

将当前 NPU-specific 实现重构为支持多硬件的通用批组合优化框架。

```
batch_composition_framework/
├── include/BatchComp/           # C++ MLIR 头文件
│   ├── BatchCompDialect.h       # 通用Dialect定义
│   ├── BatchCompOps.h            # 操作定义
│   ├── BatchCompOps.td           # TableGen定义
│   ├── BatchCompPasses.h         # Pass定义
│   └── BatchCompPasses.td        # Pass TableGen
│
├── lib/                          # C++ MLIR 实现
│   ├── Dialect/
│   │   ├── BatchCompDialect.cpp
│   │   └── BatchCompOps.cpp
│   ├── Transforms/
│   │   ├── TileSchedulerPass.cpp    # 主Pass
│   │   ├── TileGenerator.cpp        # Tile生成逻辑
│   │   └── BatchOptimizer.cpp       # 批优化接口
│   └── Hardware/
│       ├── HardwareAdapter.cpp      # 硬件适配层
│       ├── NPUBackend.cpp           # NPU后端
│       ├── GPUBackend.cpp           # GPU后端
│       └── TPUBackend.cpp           # TPU后端
│
├── python/                       # Python优化器
│   ├── core/
│   │   ├── __init__.py
│   │   ├── batch_optimizer.py       # 核心优化器基类
│   │   ├── tile_info.py            # Tile数据结构
│   │   └── hardware_config.py      # 硬件配置
│   ├── optimizers/
│   │   ├── __init__.py
│   │   ├── greedy.py               # 贪心算法
│   │   ├── improved_greedy.py      # 改进贪心
│   │   └── cpsat_optimizer.py      # CP-SAT优化
│   ├── hardware/
│   │   ├── __init__.py
│   │   ├── npu_config.py           # NPU配置
│   │   ├── gpu_config.py           # GPU配置
│   │   └── tpu_config.py           # TPU配置
│   └── cli/
│       ├── __init__.py
│       └── batch_comp_cli.py       # 命令行工具
│
├── configs/                      # 硬件配置文件
│   ├── npu.yaml
│   ├── gpu.yaml
│   └── tpu.yaml
│
├── tests/                        # 测试套件
│   ├── unit/
│   │   ├── test_optimizers.py
│   │   ├── test_tile_generation.py
│   │   └── test_hardware_configs.py
│   ├── integration/
│   │   ├── test_mlir_lowering.py
│   │   └── test_e2e_matmul.py
│   └── benchmarks/
│       ├── benchmark_suite.py
│       └── performance_comparison.py
│
├── docs/                         # 文档
│   ├── design/
│   │   └── ARCHITECTURE.md
│   ├── api/
│   │   ├── python_api.md
│   │   └── cpp_api.md
│   └── tutorials/
│       ├── quickstart.md
│       └── adding_new_hardware.md
│
├── CMakeLists.txt               # C++构建配置
├── setup.py                     # Python包配置
├── requirements.txt             # Python依赖
└── README.md                    # 项目概述
```

## 2. 重构步骤

### Phase 1: 基础重组（Week 1）
- [x] 创建新的目录结构
- [ ] 移动现有文件到新位置
- [ ] 更新import路径和include路径
- [ ] 确保现有功能不破坏

### Phase 2: MLIR Dialect重构（Week 2）
- [ ] 重命名 NPU Dialect → BatchComp Dialect
- [ ] 更新操作定义（添加slice, pad, accumulate）
- [ ] 实现硬件参数化
- [ ] 更新Pass以支持多硬件

### Phase 3: Python重构（Week 3）
- [ ] 整合现有Python代码
- [ ] 实现统一的优化器接口
- [ ] 添加硬件配置系统
- [ ] 创建Python包结构

### Phase 4: 硬件后端（Week 4）
- [ ] 抽象硬件接口
- [ ] 实现NPU后端（保持现有功能）
- [ ] 添加GPU后端支持
- [ ] 添加TPU后端支持（模拟）

### Phase 5: 测试和文档（Week 5）
- [ ] 迁移现有测试
- [ ] 添加新的单元测试
- [ ] 创建集成测试
- [ ] 更新所有文档

## 3. 关键变更

### 3.1 MLIR Dialect变更

**从NPU-specific：**
```mlir
npu.create_batch
npu.add_to_batch
npu.execute_batch
```

**到通用批组合：**
```mlir
batch_comp.slice        // 切分矩阵
batch_comp.pad          // 边界填充
batch_comp.tile_matmul  // tile矩阵乘法
batch_comp.accumulate   // K维累加
batch_comp.create_batch // 创建批次（硬件参数化）
batch_comp.execute_batch // 执行批次
```

### 3.2 Python接口统一

```python
# 统一的优化器接口
from batch_comp import UnifiedOptimizer, HardwareConfig

# 创建硬件配置
config = HardwareConfig.from_yaml("configs/npu.yaml")
# 或使用预设
config = HardwareConfig.get_default("npu")

# 创建优化器
optimizer = UnifiedOptimizer(config)

# 优化
tiles = generate_tiles(m=100, n=200, k=300)
result = optimizer.optimize(tiles, algorithm="auto")
```

### 3.3 硬件配置系统

```yaml
# configs/npu.yaml
hardware:
  type: "npu"
  vendor: "custom"

compute:
  batch_size: 16
  tile_size: [16, 16, 16]
  must_fill_batch: true
  supports_variable_tiles: true

memory:
  on_chip_buffer: 16384  # 16KB
  dma_channels: 2

optimization:
  default_algorithm: "cp-sat"
  cp_sat_time_limit: 10.0
  use_improved_greedy_threshold: 1000
```

## 4. 保持向后兼容

### 4.1 维护旧接口
- 保留 `NPUTileSchedulerPass` 作为 `BatchCompTileSchedulerPass` 的别名
- 保留旧的Python脚本，内部调用新实现

### 4.2 迁移指南
```python
# 旧代码
from CP_SAT_TILE_SCHEDULER import NPUTileSchedulerCPSAT

# 新代码
from batch_comp.optimizers import CPSATOptimizer
# 或使用兼容层
from batch_comp.legacy import NPUTileSchedulerCPSAT
```

## 5. 测试策略

### 5.1 单元测试
- 每个优化器算法
- 每个硬件配置
- Tile生成逻辑
- MLIR降低逻辑

### 5.2 集成测试
- 端到端矩阵乘法
- 不同硬件配置
- 大规模矩阵
- 性能回归测试

### 5.3 基准测试
```python
# 标准基准测试套件
benchmarks = [
    ("small", 64, 64, 64),      # 小矩阵
    ("medium", 512, 512, 512),  # 中等矩阵
    ("large", 2048, 2048, 2048), # 大矩阵
    ("irregular", 100, 200, 300), # 不规则
]

for name, m, n, k in benchmarks:
    for hardware in ["npu", "gpu", "tpu"]:
        run_benchmark(name, m, n, k, hardware)
```

## 6. 文档更新

### 6.1 需要创建的文档
- [ ] API参考文档
- [ ] 架构设计文档
- [ ] 硬件添加指南
- [ ] 性能调优指南

### 6.2 需要更新的文档
- [ ] README.md - 项目概述
- [ ] QUICKSTART.md - 快速开始
- [ ] 集成指南

## 7. 风险和缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 破坏现有功能 | 高 | 保持向后兼容层 |
| 重构时间过长 | 中 | 分阶段进行 |
| 测试覆盖不足 | 中 | 先写测试再重构 |
| 文档不同步 | 低 | 代码和文档同步更新 |

## 8. 成功标准

- [ ] 所有现有测试通过
- [ ] 新架构支持至少2种硬件
- [ ] 性能不低于当前实现
- [ ] 代码覆盖率 > 80%
- [ ] 文档完整性 > 90%

## 9. 时间线

- **Week 1**: 基础重组
- **Week 2**: MLIR重构
- **Week 3**: Python重构
- **Week 4**: 硬件后端
- **Week 5**: 测试和文档
- **Week 6**: 性能优化和发布

---

**开始日期**: 2025-11-15
**预计完成**: 2025-12-27
**负责人**: NPU Compiler Team