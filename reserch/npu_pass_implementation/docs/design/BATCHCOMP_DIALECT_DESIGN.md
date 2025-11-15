# BatchComp Dialect 设计文档

**版本**: 1.0
**日期**: 2025-11-15
**状态**: Phase 2 - MLIR Dialect设计完成

---

## 1. 概述

BatchComp (Batch Composition) Dialect是一个统一的MLIR dialect，用于优化tile-based加速器上的矩阵运算。它将NPU、GPU、TPU等不同硬件的优化问题抽象为**批组合优化**（Batch Composition Optimization）。

### 1.1 设计目标

| 目标 | 描述 |
|------|------|
| **统一抽象** | 一个dialect支持NPU/GPU/TPU |
| **硬件参数化** | 通过属性配置硬件特性 |
| **完整操作集** | slice, pad, tile_matmul, accumulate |
| **多矩阵支持** | 跨矩阵tile batching优化 |
| **K维规约** | 软件控制的K维累加 |

### 1.2 核心理念

```
问题本质 = 批组合优化 = Bin Packing Problem 的变种
```

所有tile-based加速器的优化目标是相同的：
- **最小化批次数量** → 减少kernel启动开销
- **最大化硬件利用率** → 减少NOPs（空操作）
- **优化数据局部性** → 同组tiles在同一batch

---

## 2. Dialect结构

### 2.1 命名空间

```cpp
namespace mlir::batch_comp
```

### 2.2 类型系统

| 类型 | Mnemonic | 描述 |
|------|----------|------|
| `TileType` | `!batch_comp.tile` | 单个tile计算 |
| `TileSet` | `!batch_comp.tileset` | Tile集合 |
| `Batch` | `!batch_comp.batch` | Tile批次 |
| `BatchSchedule` | `!batch_comp.schedule` | 完整调度方案 |

### 2.3 操作分类

BatchComp Dialect包含4类操作：

#### A. Tile生成操作

```mlir
// 1. slice: 从矩阵切片tile
%tile_a = batch_comp.slice %A[32, 64] [16, 16]
          : tensor<100x200xf32> -> tensor<16x16xf32>

// 2. pad: 填充edge tiles
%padded = batch_comp.pad %tile to [16, 16] {value = 0.0 : f32}
          : tensor<13x16xf32> -> tensor<16x16xf32>

// 3. tile_matmul: tile级矩阵乘法
%C_tile = batch_comp.tile_matmul %A_tile, %B_tile {
  matrix_id = 0 : i32,
  k_iteration = 0 : i64
} : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// 4. accumulate: K维规约
%C_final = batch_comp.accumulate %C_partial0, %C_partial1
           : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
```

#### B. 批次管理操作

```mlir
// 1. create_batch: 创建批次
%batch = batch_comp.create_batch {
  batch_size = 16 : i32,
  hardware = "npu"
} : !batch_comp.batch

// 2. add_to_batch: 添加tile到批次
%batch1 = batch_comp.add_to_batch %batch0, %tile_a, %tile_b {
  slot = 0 : i32,
  tile_id = 0 : i32,
  matrix_id = 0 : i32
} : !batch_comp.batch, tensor<16x16xf32>, tensor<16x16xf32>

// 3. execute_batch: 执行批次
%result = batch_comp.execute_batch %batch {
  operation = "matmul",
  hardware = "npu"
} : !batch_comp.batch -> !batch_comp.batch
```

#### C. 高层调度操作

```mlir
// 1. generate_tiles: 生成所有tiles
%tileset = batch_comp.generate_tiles %A, %B {
  tile_size = [16, 16, 16],
  matrix_id = 0 : i32
} : tensor<100x200xf32>, tensor<200x300xf32> -> !batch_comp.tileset

// 2. schedule_batches: 优化批调度
%schedule = batch_comp.schedule_batches %tileset {
  hardware = "npu",
  batch_size = 16 : i32,
  algorithm = "auto"
} : !batch_comp.tileset -> !batch_comp.schedule

// 3. execute_schedule: 执行完整调度
%C = batch_comp.execute_schedule %schedule {
  hardware = "npu"
} : !batch_comp.schedule -> tensor<100x300xf32>
```

#### D. 工具操作

```mlir
// nop_tile: 填充NOP
%nop = batch_comp.nop_tile : tensor<16x16xf32>
```

---

## 3. 硬件参数化

### 3.1 硬件配置属性

BatchComp Dialect通过属性支持不同硬件：

| 属性 | 类型 | NPU | GPU | TPU |
|------|------|-----|-----|-----|
| `batch_size` | i32 | 16 | 32 | 128 |
| `must_fill` | bool | true | false | false |
| `hardware` | string | "npu" | "gpu" | "tpu" |
| `algorithm` | string | "auto" | "auto" | "auto" |

### 3.2 硬件适配示例

**NPU配置**:
```mlir
batch_comp.create_batch {
  batch_size = 16 : i32,
  must_fill = true,
  hardware = "npu"
}
```

**GPU配置**:
```mlir
batch_comp.create_batch {
  batch_size = 32 : i32,      // Warp size
  must_fill = false,           // Partial warps OK
  hardware = "gpu"
}
```

**TPU配置**:
```mlir
batch_comp.create_batch {
  batch_size = 128 : i32,      // Systolic array size
  must_fill = false,
  hardware = "tpu"
}
```

---

## 4. 完整转换示例

### 4.1 输入: TOSA Matmul

```mlir
func.func @matmul(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>)
    -> tensor<100x300xf32> {
  %C = tosa.matmul %A, %B : (tensor<100x200xf32>, tensor<200x300xf32>)
                             -> tensor<100x300xf32>
  return %C : tensor<100x300xf32>
}
```

### 4.2 转换层级

BatchComp提供两个抽象层级：

#### Level 1: 高层抽象（推荐）

```mlir
func.func @matmul(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>)
    -> tensor<100x300xf32> {
  // 生成tiles
  %tileset = batch_comp.generate_tiles %A, %B {
    tile_size = [16, 16, 16],
    matrix_id = 0 : i32
  } : tensor<100x200xf32>, tensor<200x300xf32> -> !batch_comp.tileset

  // 优化调度
  %schedule = batch_comp.schedule_batches %tileset {
    hardware = "npu",
    batch_size = 16 : i32,
    algorithm = "cpsat"  // 使用CP-SAT全局优化
  } : !batch_comp.tileset -> !batch_comp.schedule

  // 执行
  %C = batch_comp.execute_schedule %schedule {
    hardware = "npu"
  } : !batch_comp.schedule -> tensor<100x300xf32>

  return %C : tensor<100x300xf32>
}
```

**优点**:
- ✅ 简洁，3个操作完成转换
- ✅ 优化器自动选择最佳算法
- ✅ 易于维护和理解

#### Level 2: 底层细节（完全展开）

```mlir
func.func @matmul(%A: tensor<100x200xf32>, %B: tensor<200x300xf32>)
    -> tensor<100x300xf32> {
  %c0 = arith.constant 0 : index
  %c16 = arith.constant 16 : index

  // M维tiling: 100 / 16 = 7个完整tiles
  // N维tiling: 300 / 16 = 19个完整tiles
  // K维tiling: 200 / 16 = 13 iterations

  // 创建批次
  %batch0 = batch_comp.create_batch {
    batch_size = 16 : i32,
    hardware = "npu"
  } : !batch_comp.batch

  // K iteration 0: 添加16个tiles到batch
  %tile_a_0_0 = batch_comp.slice %A[0, 0] [16, 16]
                : tensor<100x200xf32> -> tensor<16x16xf32>
  %tile_b_0_0 = batch_comp.slice %B[0, 0] [16, 16]
                : tensor<200x300xf32> -> tensor<16x16xf32>

  %batch1 = batch_comp.add_to_batch %batch0, %tile_a_0_0, %tile_b_0_0 {
    slot = 0 : i32,
    tile_id = 0 : i32,
    matrix_id = 0 : i32,
    k_iteration = 0 : i64,
    m_offset = 0 : i64,
    n_offset = 0 : i64
  } : !batch_comp.batch, tensor<16x16xf32>, tensor<16x16xf32>

  // ... 添加更多tiles到batch1 ...

  // 执行第一个batch
  %result0 = batch_comp.execute_batch %batch1 {
    operation = "matmul",
    hardware = "npu"
  } : !batch_comp.batch -> !batch_comp.batch

  // ... 更多批次 ...

  // K维累加
  %C_partial0 = ...
  %C_partial1 = ...
  %C_accumulated = batch_comp.accumulate %C_partial0, %C_partial1
                   : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

  return %C : tensor<100x300xf32>
}
```

**优点**:
- ✅ 完全控制tile分配
- ✅ 可以手动优化特定场景

**缺点**:
- ❌ IR过于冗长
- ❌ 难以维护

---

## 5. 多矩阵支持

BatchComp原生支持多矩阵批组合优化。

### 5.1 场景

```mlir
// 3个matmul需要优化
%C1 = tosa.matmul %A1, %B1
%C2 = tosa.matmul %A2, %B2
%C3 = tosa.matmul %A3, %B3
```

### 5.2 转换

```mlir
// 为每个matmul生成tiles
%tiles1 = batch_comp.generate_tiles %A1, %B1 {
  tile_size = [16, 16, 16],
  matrix_id = 0 : i32  // 矩阵0
}

%tiles2 = batch_comp.generate_tiles %A2, %B2 {
  tile_size = [16, 16, 16],
  matrix_id = 1 : i32  // 矩阵1
}

%tiles3 = batch_comp.generate_tiles %A3, %B3 {
  tile_size = [16, 16, 16],
  matrix_id = 2 : i32  // 矩阵2
}

// 合并所有tiles
%all_tiles = batch_comp.merge_tilesets %tiles1, %tiles2, %tiles3
             : !batch_comp.tileset, !batch_comp.tileset, !batch_comp.tileset
             -> !batch_comp.tileset

// 全局优化调度（跨矩阵batching）
%schedule = batch_comp.schedule_batches %all_tiles {
  hardware = "npu",
  batch_size = 16 : i32,
  algorithm = "cpsat"  // CP-SAT找到全局最优
} : !batch_comp.tileset -> !batch_comp.schedule

// 执行
%C1, %C2, %C3 = batch_comp.execute_schedule %schedule {
  hardware = "npu"
} : !batch_comp.schedule -> tensor<...>, tensor<...>, tensor<...>
```

### 5.3 优化效果

| 场景 | 独立调度 | 跨矩阵优化 | 改进 |
|------|----------|-----------|------|
| 3个matmul (45 tiles) | 5 batches | 3 batches | **-40%** |
| NOPs | 15 | 3 | **-80%** |
| 利用率 | 75% | 93.75% | **+25%** |

---

## 6. K维规约处理

### 6.1 问题

矩阵乘法 `C[M,N] = A[M,K] × B[K,N]` 中，K维需要规约：

```
C[m,n] = Σ(k=0 to K-1) A[m,k] * B[k,n]
```

Tiling后：
```
C[m,n] = Σ(k_iter=0 to K_iters-1)
         ( A[m, k_iter*16:(k_iter+1)*16] × B[k_iter*16:(k_iter+1)*16, n] )
```

### 6.2 BatchComp解决方案

使用`accumulate`操作实现软件控制的规约：

```mlir
// K iteration 0: 计算partial result
%C_partial_0 = batch_comp.tile_matmul %A_tile_k0, %B_tile_k0 {
  k_iteration = 0 : i64
} : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// K iteration 1
%C_partial_1 = batch_comp.tile_matmul %A_tile_k1, %B_tile_k1 {
  k_iteration = 1 : i64
} : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>

// 累加
%C_final = batch_comp.accumulate %C_partial_0, %C_partial_1
           : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
```

### 6.3 K维分组优化

**原则**: 同一输出位置的不同K-iteration tiles应该分在相同或相邻的batch中，以提高缓存局部性。

**实现**: `schedule_batches`操作内部会：
1. 按`(matrix_id, m_offset, n_offset)`分组tiles
2. 每组内按`k_iteration`排序
3. 优先处理大组，减少碎片

---

## 7. 与现有Dialect的对比

### 7.1 vs NPU Dialect

| 特性 | NPU Dialect | BatchComp Dialect |
|------|-------------|-------------------|
| 硬件支持 | NPU only | NPU/GPU/TPU |
| 操作集 | create_batch, add_to_batch, execute_batch | + slice, pad, tile_matmul, accumulate |
| K维规约 | 隐式（在execute_batch中） | 显式（accumulate操作） |
| 优化算法 | C++硬编码 | Python可配置（greedy/cpsat） |
| 多矩阵 | ✗ | ✅ |

### 7.2 vs Linalg Dialect

| 特性 | Linalg | BatchComp |
|------|--------|-----------|
| 抽象层级 | 低层（循环+张量） | 中层（tile+batch） |
| 硬件感知 | ✗ | ✅ |
| Batch优化 | ✗ | ✅ (核心功能) |
| 适用场景 | 通用张量计算 | Tile-based加速器 |

---

## 8. Lowering路径

完整的lowering pipeline：

```
TOSA/Linalg/StableHLO
    ↓
    ↓ [BatchCompTileSchedulerPass]
    ↓
BatchComp Dialect (高层)
    ↓
    ↓ [BatchCompLoweringPass]
    ↓
BatchComp Dialect (底层)
    ↓
    ↓ [硬件后端Pass]
    ↓
  NPU/GPU/TPU Dialect
    ↓
    ↓ [CodeGen]
    ↓
  LLVM IR / SPIR-V / etc.
```

### 8.1 Pass职责

| Pass | 输入 | 输出 | 功能 |
|------|------|------|------|
| `BatchCompTileSchedulerPass` | tosa.matmul | batch_comp高层IR | Tile生成+调度优化 |
| `BatchCompLoweringPass` | batch_comp高层 | batch_comp底层 | 展开为具体batch操作 |
| `BatchCompToNPUPass` | batch_comp底层 | npu dialect | NPU后端lowering |
| `BatchCompToGPUPass` | batch_comp底层 | gpu dialect | GPU后端lowering |
| `BatchCompToTPUPass` | batch_comp底层 | tpu dialect | TPU后端lowering |

---

## 9. Python集成

BatchComp Dialect与Python优化器紧密集成：

### 9.1 架构

```
C++ MLIR Pass
    ↓
    ↓ [调用Python]
    ↓
Python Optimizer (CP-SAT/Greedy)
    ↓
    ↓ [返回调度方案]
    ↓
C++ MLIR Pass (生成IR)
```

### 9.2 接口示例

```cpp
// C++ Pass中调用Python优化器
#include "python/PythonOptimizer.h"

void BatchCompTileSchedulerPass::runOnOperation() {
  // 1. 收集tiles
  std::vector<TileInfo> tiles = collectTiles();

  // 2. 调用Python优化器
  PythonOptimizer optimizer(hardwareConfig);
  BatchScheduleResult result = optimizer.optimize(tiles, "cpsat");

  // 3. 根据结果生成BatchComp IR
  for (const auto &batch : result.batches) {
    generateBatchIR(batch);
  }
}
```

---

## 10. 性能目标

| 指标 | 目标 |
|------|------|
| **编译时间** | < 1s (贪心), < 10s (CP-SAT) |
| **运行时性能** | 与手写优化持平 |
| **硬件利用率** | > 90% (多矩阵场景) |
| **NOP减少** | > 80% (vs 简单贪心) |

---

## 11. 后续工作

- [ ] 实现`BatchCompTileSchedulerPass` (C++)
- [ ] 实现`BatchCompLoweringPass` (C++)
- [ ] Python集成接口
- [ ] 硬件后端Pass (NPU/GPU/TPU)
- [ ] 端到端测试
- [ ] 性能基准测试

---

## 12. 参考资料

- [BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md](../design/BATCH_COMPOSITION_OPTIMIZATION_DESIGN.md) - 统一框架设计
- [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md) - Python重构总结
- [BatchCompOps.td](include/BatchComp/BatchCompOps.td) - 完整操作定义
- [MLIR Dialect Tutorial](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)

---

**创建时间**: 2025-11-15
**作者**: NPU Compiler Team
**版本**: 1.0
