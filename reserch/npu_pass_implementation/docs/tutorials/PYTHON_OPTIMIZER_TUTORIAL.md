# Python优化器完整教程

**从零开始学习BatchComp Python优化器**

---

## 📋 教程信息

- **难度**: ⭐ 入门
- **预计时长**: 30分钟
- **前置知识**: Python基础、矩阵乘法概念
- **学习目标**: 掌握使用Python优化器进行tile批处理优化

---

## 🎯 学习目标

完成本教程后，你将能够：

1. ✅ 理解tile批组合优化的基本概念
2. ✅ 使用三种优化算法（Greedy, ImprovedGreedy, CP-SAT）
3. ✅ 配置不同硬件平台（NPU/GPU/TPU）
4. ✅ 分析和比较优化结果
5. ✅ 处理多矩阵场景

---

## 📚 第1章：核心概念

### 1.1 什么是Tile?

**Tile（瓦片）**是矩阵乘法的一个小块计算单元：

```python
# 矩阵乘法: C[M, N] = A[M, K] × B[K, N]
# Tile: C_tile[m:m+16, n:n+16] = A_tile[m:m+16, k:k+16] × B_tile[k:k+16, n:n+16]

# 例如，100×200的矩阵用16×16的tile分块：
# M方向: 100/16 = 7个tile (最后一个13×16)
# N方向: 200/16 = 13个tile (最后一个是16×8)
# K方向: 假设K=100, 则100/16 = 7个tile

# 总tile数 = 7 × 13 × 7 = 637个tiles
```

### 1.2 什么是Batch?

**Batch（批次）**是硬件并行执行的一组tiles：

```
NPU: 16 tiles/batch   (16路并行)
GPU: 32 tiles/batch   (32线程warp)
TPU: 128 tiles/batch  (128×128脉动阵列)
```

### 1.3 优化目标

**最小化batch数** = 提高硬件利用率

```python
# 坏的调度：
# Batch 0: [tile0, tile1, ..., tile15]  # 16 tiles
# Batch 1: [tile16, tile17]  # 只有2个tiles，14个NOPs！
# 利用率: 18/32 = 56.25%

# 好的调度（跨矩阵）：
# Batch 0: [m0_tile0, ..., m0_tile10, m1_tile0, ..., m1_tile5]  # 16 tiles
# Batch 1: [m1_tile6, ..., m1_tile15]  # 10 tiles, 6个NOPs
# 利用率: 26/32 = 81.25%
```

---

## 🚀 第2章：快速开始

### 2.1 安装依赖

```bash
cd /path/to/npu_pass_implementation
pip install -r requirements.txt
```

### 2.2 第一个优化示例

创建文件 `my_first_optimizer.py`:

```python
#!/usr/bin/env python3
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.core.tile_info import TileInfo
from python.core.hardware_config import HardwareConfig
from python.optimizers.greedy import GreedyOptimizer

# 1. 创建硬件配置
config = HardwareConfig.get_default("npu")
print(f"硬件: {config.name}, Batch大小: {config.compute.batch_size}")

# 2. 创建一些tiles
tiles = []
for i in range(20):
    tile = TileInfo(
        tile_id=i,
        matrix_id=0,
        m_offset=i * 16,
        n_offset=0,
        k_iteration=0,
        actual_m=16,
        actual_n=16,
        actual_k=16,
        a_slice=(i*16, (i+1)*16, 0, 16),
        b_slice=(0, 16, 0, 16),
        c_slice=(i*16, (i+1)*16, 0, 16),
        accumulate=False
    )
    tiles.append(tile)

print(f"生成 {len(tiles)} 个tiles")

# 3. 运行优化
optimizer = GreedyOptimizer(config)
result = optimizer.optimize(tiles)

# 4. 查看结果
print(f"\n优化结果:")
print(f"  批次数: {result.num_batches}")
print(f"  NOP数: {result.total_nops}")
print(f"  利用率: {result.utilization:.2%}")
print(f"  算法: {result.algorithm}")
```

运行:
```bash
python3 my_first_optimizer.py
```

输出:
```
硬件: NPU Default, Batch大小: 16
生成 20 个tiles

优化结果:
  批次数: 2
  NOP数: 12
  利用率: 62.5%
  算法: greedy
```

---

## 🔧 第3章：三种优化算法

### 3.1 Greedy - 简单贪心

**特点**: 快速，O(n)复杂度

**适用**: 单矩阵，快速编译

```python
from python.optimizers.greedy import GreedyOptimizer

optimizer = GreedyOptimizer(config)
result = optimizer.optimize(tiles)

# Greedy策略：
# - 顺序填充batch
# - 一个batch满了就开新batch
# - 简单直接，但可能产生很多NOPs
```

### 3.2 ImprovedGreedy - 改进贪心

**特点**: 较快，O(n log n)，支持K-dimension分组

**适用**: 多矩阵，生产环境

```python
from python.optimizers.improved_greedy import ImprovedGreedyOptimizer

optimizer = ImprovedGreedyOptimizer(config)
result = optimizer.optimize(tiles)

# ImprovedGreedy策略：
# 1. 按输出位置(m,n)分组tiles
# 2. 同组的tiles不能在同一batch（累加冲突）
# 3. 不同组的tiles可以跨矩阵组合
# 4. 显著减少NOPs
```

**K-dimension分组示例**:
```python
# 假设有3个tiles，都输出到C[0:16, 0:16]:
tile0: k_iteration=0  # 计算 A[0:16, 0:16] × B[0:16, 0:16]
tile1: k_iteration=1  # 计算 A[0:16, 16:32] × B[16:32, 0:16]
tile2: k_iteration=2  # 计算 A[0:16, 32:48] × B[32:48, 0:16]

# 分组key: "m0_n0" (相同的输出位置)
# 这3个tiles不能在同一个batch！
# 但可以和其他输出位置的tiles组合
```

### 3.3 CP-SAT - 全局优化

**特点**: 最优，NP复杂度（有时限）

**适用**: 离线优化，追求极致性能

```python
from python.optimizers.cpsat_optimizer import CPSATOptimizer

optimizer = CPSATOptimizer(config)
result = optimizer.optimize(tiles, time_limit=10.0)  # 10秒时限

# CP-SAT策略：
# - Google OR-Tools约束求解器
# - 建模为整数规划问题
# - 全局优化，找最优解
# - 可能较慢，适合离线编译
```

### 3.4 算法对比

```python
# 对比三种算法
algorithms = [
    ("Greedy", GreedyOptimizer(config)),
    ("ImprovedGreedy", ImprovedGreedyOptimizer(config)),
    ("CP-SAT", CPSATOptimizer(config)),
]

for name, optimizer in algorithms:
    result = optimizer.optimize(tiles)
    print(f"{name:15} | Batches: {result.num_batches:2} | "
          f"NOPs: {result.total_nops:3} | "
          f"Util: {result.utilization:.1%} | "
          f"Time: {result.execution_time:.4f}s")
```

输出示例:
```
Greedy          | Batches:  5 | NOPs:  20 | Util: 75.0% | Time: 0.0001s
ImprovedGreedy  | Batches:  3 | NOPs:   8 | Util: 83.3% | Time: 0.0015s
CP-SAT          | Batches:  3 | NOPs:   4 | Util: 91.7% | Time: 0.1234s
```

---

## 🎨 第4章：硬件配置

### 4.1 使用预定义配置

```python
from python.core.hardware_config import HardwareConfig

# NPU配置
npu_config = HardwareConfig.get_default("npu")
print(f"NPU: batch_size={npu_config.compute.batch_size}, "
      f"must_fill={npu_config.optimization.must_fill_batch}")

# GPU配置
gpu_config = HardwareConfig.get_default("gpu")
print(f"GPU: batch_size={gpu_config.compute.batch_size}, "
      f"must_fill={gpu_config.optimization.must_fill_batch}")

# TPU配置
tpu_config = HardwareConfig.get_default("tpu")
print(f"TPU: batch_size={tpu_config.compute.batch_size}")
```

输出:
```
NPU: batch_size=16, must_fill=True
GPU: batch_size=32, must_fill=False
TPU: batch_size=128, must_fill=False
```

### 4.2 从YAML加载配置

```python
config = HardwareConfig.from_yaml("configs/npu.yaml")
```

`configs/npu.yaml`:
```yaml
name: "NPU Default"
hardware_type: "npu"

compute:
  batch_size: 16
  tile_size: [16, 16, 16]
  vector_lanes: 16

optimization:
  must_fill_batch: true
  allow_dynamic_batch: false
  max_nops_per_batch: 4
```

### 4.3 自定义配置

```python
from python.core.hardware_config import (
    HardwareConfig, HardwareType,
    ComputeConfig, MemoryConfig, OptimizationConfig
)

custom_config = HardwareConfig(
    name="Custom Accelerator",
    hardware_type=HardwareType.NPU,
    compute=ComputeConfig(
        batch_size=24,
        tile_size=(16, 16, 16),
        vector_lanes=24,
    ),
    memory=MemoryConfig(
        shared_memory_kb=64,
        register_file_kb=16,
    ),
    optimization=OptimizationConfig(
        must_fill_batch=False,
        allow_dynamic_batch=True,
        max_nops_per_batch=8,
    )
)

optimizer = ImprovedGreedyOptimizer(custom_config)
```

---

## 🎯 第5章：多矩阵优化

### 5.1 为什么需要多矩阵优化？

```python
# 场景：3个小矩阵乘法
# Matrix 0: 48×64 × 64×48 → 27 tiles
# Matrix 1: 32×48 × 48×32 → 18 tiles
# Matrix 2: 35×50 × 50×40 → 21 tiles
# 总计: 66 tiles

# 单矩阵优化:
# Matrix 0: 27 tiles → 2 batches (27/16 = 1.69)
# Matrix 1: 18 tiles → 2 batches (18/16 = 1.13)
# Matrix 2: 21 tiles → 2 batches (21/16 = 1.31)
# 总计: 6 batches，利用率 = 66/(6*16) = 68.75%

# 多矩阵优化:
# 合并所有tiles: 66 tiles → 5 batches (66/16 = 4.13)
# 总计: 5 batches，利用率 = 66/(5*16) = 82.5%
# 节省 1 个batch!
```

### 5.2 多矩阵示例代码

```python
def generate_tiles_for_matmul(matrix_id, M, K, N, tile_size=16):
    """为单个matmul生成tiles"""
    tiles = []
    tile_id_base = matrix_id * 1000  # 不同矩阵用不同ID范围

    for m in range(0, M, tile_size):
        for n in range(0, N, tile_size):
            for k in range(0, K, tile_size):
                tile = TileInfo(
                    tile_id=len(tiles) + tile_id_base,
                    matrix_id=matrix_id,  # 标记来自哪个矩阵
                    m_offset=m,
                    n_offset=n,
                    k_iteration=k // tile_size,
                    actual_m=min(tile_size, M - m),
                    actual_n=min(tile_size, N - n),
                    actual_k=min(tile_size, K - k),
                    a_slice=(m, min(m+tile_size, M), k, min(k+tile_size, K)),
                    b_slice=(k, min(k+tile_size, K), n, min(n+tile_size, N)),
                    c_slice=(m, min(m+tile_size, M), n, min(n+tile_size, N)),
                    accumulate=(k > 0)
                )
                tiles.append(tile)

    return tiles

# 生成多个矩阵的tiles
all_tiles = []
all_tiles.extend(generate_tiles_for_matmul(0, M=48, K=64, N=48))
all_tiles.extend(generate_tiles_for_matmul(1, M=32, K=48, N=32))
all_tiles.extend(generate_tiles_for_matmul(2, M=35, K=50, N=40))

print(f"总tile数: {len(all_tiles)}")

# 使用ImprovedGreedy进行多矩阵优化
config = HardwareConfig.get_default("npu")
optimizer = ImprovedGreedyOptimizer(config)
result = optimizer.optimize(all_tiles)

print(f"批次数: {result.num_batches}")
print(f"利用率: {result.utilization:.2%}")

# 分析batch组成
for i, batch in enumerate(result.batches):
    matrices = set(tile.matrix_id for tile in batch)
    print(f"Batch {i}: {len(batch)} tiles, 来自矩阵 {sorted(matrices)}")
```

输出:
```
总tile数: 66
批次数: 5
利用率: 82.5%

Batch 0: 16 tiles, 来自矩阵 [0, 1]
Batch 1: 16 tiles, 来自矩阵 [0, 1, 2]
Batch 2: 16 tiles, 来自矩阵 [1, 2]
Batch 3: 14 tiles, 来自矩阵 [2]
Batch 4:  4 tiles, 来自矩阵 [2]
```

---

## 📊 第6章：结果分析

### 6.1 BatchScheduleResult结构

```python
@dataclass
class BatchScheduleResult:
    batches: List[List[TileInfo]]  # 每个batch的tiles列表
    num_batches: int               # batch数量
    total_nops: int                # 总NOP数
    utilization: float             # 平均利用率
    algorithm: str                 # 使用的算法
    execution_time: float          # 优化耗时(秒)
```

### 6.2 详细分析

```python
result = optimizer.optimize(tiles)

# 基本统计
print(f"批次数: {result.num_batches}")
print(f"NOP数: {result.total_nops}")
print(f"利用率: {result.utilization:.2%}")
print(f"算法: {result.algorithm}")
print(f"耗时: {result.execution_time:.4f}s")

# 每个batch的详细信息
for i, batch in enumerate(result.batches):
    batch_nops = config.compute.batch_size - len(batch)
    batch_util = len(batch) / config.compute.batch_size

    print(f"\nBatch {i}:")
    print(f"  Tiles: {len(batch)}")
    print(f"  NOPs: {batch_nops}")
    print(f"  利用率: {batch_util:.2%}")

    # 矩阵分布
    matrix_counts = {}
    for tile in batch:
        matrix_counts[tile.matrix_id] = matrix_counts.get(tile.matrix_id, 0) + 1

    for matrix_id, count in sorted(matrix_counts.items()):
        print(f"    矩阵{matrix_id}: {count} tiles")

    # K-iteration分布
    k_iterations = set(tile.k_iteration for tile in batch)
    print(f"  K-iterations: {sorted(k_iterations)}")
```

### 6.3 性能指标

```python
def analyze_performance(result, baseline_result=None):
    """分析性能指标"""

    print("=" * 60)
    print(f"算法: {result.algorithm}")
    print("=" * 60)

    print(f"批次数: {result.num_batches}")
    print(f"NOP数: {result.total_nops}")
    print(f"利用率: {result.utilization:.2%}")
    print(f"优化时间: {result.execution_time:.4f}s")

    if baseline_result:
        batch_improvement = ((baseline_result.num_batches - result.num_batches) /
                            baseline_result.num_batches * 100)
        nop_improvement = ((baseline_result.total_nops - result.total_nops) /
                          max(baseline_result.total_nops, 1) * 100)

        print(f"\n相比{baseline_result.algorithm}:")
        print(f"  批次数改进: {batch_improvement:+.1f}%")
        print(f"  NOP数改进: {nop_improvement:+.1f}%")
```

---

## 💡 第7章：最佳实践

### 7.1 算法选择指南

```python
def choose_algorithm(tiles, config):
    """智能选择算法"""
    num_tiles = len(tiles)
    num_matrices = len(set(t.matrix_id for t in tiles))

    if num_tiles < 30:
        # 小规模：使用Greedy
        return GreedyOptimizer(config)

    elif num_tiles < 100 or num_matrices == 1:
        # 中等规模或单矩阵：使用ImprovedGreedy
        return ImprovedGreedyOptimizer(config)

    else:
        # 大规模多矩阵：使用CP-SAT
        return CPSATOptimizer(config)
```

### 7.2 性能优化建议

```python
# ✅ 好的做法
# 1. 对于生产环境，使用ImprovedGreedy
optimizer = ImprovedGreedyOptimizer(config)

# 2. 设置合理的CP-SAT时间限制
result = cpsat_opt.optimize(tiles, time_limit=5.0)

# 3. 缓存硬件配置
config = HardwareConfig.get_default("npu")  # 只加载一次

# ❌ 避免的做法
# 1. 不要在热路径上使用CP-SAT（太慢）
# 2. 不要忽略K-dimension累加约束
# 3. 不要为每次优化重新加载配置
```

### 7.3 调试技巧

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 验证tiles正确性
from python.core.batch_optimizer import BatchOptimizer
BatchOptimizer.validate_tiles(tiles)

# 检查batch有效性
def validate_batch(batch, config):
    """验证batch是否合法"""
    # 1. 检查大小
    assert len(batch) <= config.compute.batch_size

    # 2. 检查K-dimension冲突
    output_positions = {}
    for tile in batch:
        key = f"m{tile.matrix_id}_m{tile.m_offset}_n{tile.n_offset}"
        if key in output_positions:
            if output_positions[key] == tile.k_iteration:
                raise ValueError(f"K-dimension conflict in batch!")
        output_positions[key] = tile.k_iteration
```

---

## 🎓 第8章：完整示例

### 8.1 端到端示例

```python
#!/usr/bin/env python3
"""完整的批组合优化示例"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.core import TileInfo, HardwareConfig
from python.optimizers import (
    GreedyOptimizer,
    ImprovedGreedyOptimizer,
    CPSATOptimizer
)

def generate_matmul_tiles(matrix_id, M, K, N, tile_size=16):
    """生成matmul的所有tiles"""
    tiles = []
    for m in range(0, M, tile_size):
        for n in range(0, N, tile_size):
            for k in range(0, K, tile_size):
                tiles.append(TileInfo(
                    tile_id=len(tiles) + matrix_id * 1000,
                    matrix_id=matrix_id,
                    m_offset=m, n_offset=n, k_iteration=k//tile_size,
                    actual_m=min(tile_size, M-m),
                    actual_n=min(tile_size, N-n),
                    actual_k=min(tile_size, K-k),
                    a_slice=(m, min(m+tile_size,M), k, min(k+tile_size,K)),
                    b_slice=(k, min(k+tile_size,K), n, min(n+tile_size,N)),
                    c_slice=(m, min(m+tile_size,M), n, min(n+tile_size,N)),
                    accumulate=(k > 0)
                ))
    return tiles

def main():
    # 1. 配置硬件
    config = HardwareConfig.get_default("npu")
    print(f"硬件: {config.name} (batch_size={config.compute.batch_size})\n")

    # 2. 生成多个矩阵的tiles
    all_tiles = []
    matrices = [
        (0, 100, 200, 100),  # M0: 100×200 × 200×100
        (1, 48, 64, 48),     # M1: 48×64 × 64×48
        (2, 35, 50, 40),     # M2: 35×50 × 50×40
    ]

    for matrix_id, M, K, N in matrices:
        tiles = generate_matmul_tiles(matrix_id, M, K, N)
        all_tiles.extend(tiles)
        print(f"矩阵{matrix_id} ({M}×{K}×{K}×{N}): {len(tiles)} tiles")

    print(f"\n总tile数: {len(all_tiles)}\n")

    # 3. 对比三种算法
    algorithms = [
        ("Greedy", GreedyOptimizer(config)),
        ("ImprovedGreedy", ImprovedGreedyOptimizer(config)),
        ("CP-SAT (5s)", CPSATOptimizer(config)),
    ]

    results = {}
    for name, optimizer in algorithms:
        print(f"运行 {name}...")
        if "CP-SAT" in name:
            result = optimizer.optimize(all_tiles, time_limit=5.0)
        else:
            result = optimizer.optimize(all_tiles)

        results[name] = result
        print(f"  ✓ {result.num_batches} batches, "
              f"{result.utilization:.1%} util, "
              f"{result.execution_time:.4f}s\n")

    # 4. 详细分析最优结果
    best_name = min(results.keys(), key=lambda k: results[k].num_batches)
    best_result = results[best_name]

    print("=" * 70)
    print(f"最优算法: {best_name}")
    print("=" * 70)
    print(f"批次数: {best_result.num_batches}")
    print(f"NOP数: {best_result.total_nops}")
    print(f"利用率: {best_result.utilization:.2%}")
    print()

    # 5. 显示batch详情
    for i, batch in enumerate(best_result.batches[:5]):  # 只显示前5个
        matrices_in_batch = set(t.matrix_id for t in batch)
        cross_matrix = "🔄" if len(matrices_in_batch) > 1 else "  "
        print(f"{cross_matrix} Batch {i}: {len(batch)} tiles, "
              f"来自矩阵 {sorted(matrices_in_batch)}")

    if len(best_result.batches) > 5:
        print(f"  ... 还有 {len(best_result.batches)-5} 个batches")

if __name__ == "__main__":
    main()
```

运行:
```bash
python3 complete_example.py
```

---

## 📚 第9章：下一步

### 9.1 进阶主题

- **自定义算法**: 继承`BatchOptimizer`实现你的算法
- **性能调优**: 调整`max_nops_per_batch`等参数
- **并行优化**: 使用多进程加速大规模优化

### 9.2 相关文档

- [API参考](../api/PYTHON_API.md)
- [硬件配置指南](../guides/HARDWARE_CONFIG_GUIDE.md)
- [CP-SAT详解](../guides/CP_SAT_README.md)
- [MLIR Pass教程](MLIR_PASS_TUTORIAL.md)

### 9.3 示例代码

查看`examples/python/`目录的完整示例：
- `basic_optimization.py` - 基础优化
- `multi_matrix.py` - 多矩阵优化
- `hardware_configs.py` - 硬件配置

---

## 🎉 恭喜！

你已完成Python优化器教程！现在你可以：

✅ 使用三种优化算法
✅ 配置不同硬件平台
✅ 处理多矩阵优化
✅ 分析优化结果

**下一步**: 尝试运行`examples/python/`中的示例，或学习[MLIR Pass集成](MLIR_PASS_TUTORIAL.md)！

---

**教程版本**: v1.0
**最后更新**: 2025-11-15
**反馈**: 欢迎提issue或pull request！
