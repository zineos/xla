# Batch Composition Optimization - 统一加速器优化框架

> **核心洞察**: 所有 tile-based 加速器的优化本质是批组合（Batch Composition）问题
> **项目状态**: 设计阶段
> **目标**: 创建通用的批组合优化框架，支持 NPU/GPU/TPU
> **最后更新**: 2025-11-15

---

## 1. 问题定义

### 1.1 核心问题

所有 tile-based 加速器都面临同一个优化问题：

```
输入：N 个独立的计算任务（tiles）
约束：硬件的批处理规格
输出：最优的批次划分
目标：最小化执行时间和资源浪费
```

### 1.2 不同硬件的表现形式

| 硬件 | 批大小 | 约束 | 填充策略 | 优化目标 |
|------|--------|------|----------|----------|
| **NPU** | 16 tiles | 必须填满 | NOP指令 | 最小化NOP |
| **GPU** | 32 threads (warp) | 可部分激活 | Predicate mask | 最大化占用率 |
| **TPU** | 128×128 systolic | 可padding | 零填充 | 最小化padding |

### 1.3 为什么这是正确的抽象

- **问题统一**: 装箱问题（Bin Packing）的变种
- **算法通用**: 约束满足问题（CSP）有成熟解法
- **优化目标一致**: 最大化硬件利用率

---

## 2. MLIR 转换设计

### 2.1 输入 MLIR

支持标准的 ML 框架降低后的表示：

```mlir
// 输入格式 1: TOSA Dialect
module @input_tosa {
  func.func @matmul(%A: tensor<100x200xf32>,
                    %B: tensor<200x300xf32>) -> tensor<100x300xf32> {
    %0 = tosa.matmul %A, %B :
         (tensor<100x200xf32>, tensor<200x300xf32>) -> tensor<100x300xf32>
    return %0 : tensor<100x300xf32>
  }
}

// 输入格式 2: Linalg Dialect
module @input_linalg {
  func.func @matmul(%A: memref<100x200xf32>,
                    %B: memref<200x300xf32>,
                    %C: memref<100x300xf32>) {
    linalg.matmul ins(%A, %B) outs(%C)
    return
  }
}

// 输入格式 3: StableHLO (from XLA)
module @input_stablehlo {
  func.func @matmul(%A: tensor<100x200xf32>,
                    %B: tensor<200x300xf32>) -> tensor<100x300xf32> {
    %0 = stablehlo.dot %A, %B
    return %0 : tensor<100x300xf32>
  }
}
```

### 2.2 Batch Composition Dialect 定义

```tablegen
// 核心 Dialect - 批组合优化
def BatchComp_Dialect : Dialect {
  let name = "batch_comp";
  let summary = "Batch composition optimization for tile-based accelerators";
  let description = [{
    统一的批组合优化框架，将大矩阵运算分解为硬件友好的批次执行。
    支持 NPU (固定批), GPU (warp批), TPU (systolic批)。
  }];
  let cppNamespace = "::mlir::batch_comp";
}

//===----------------------------------------------------------------------===//
// 基础操作
//===----------------------------------------------------------------------===//

// 1. Slice - 从大张量切出tile
def BatchComp_SliceOp : BatchComp_Op<"slice", [Pure]> {
  let summary = "Extract a tile from a tensor";
  let arguments = (ins
    AnyRankedTensor:$source,
    I64ArrayAttr:$offsets,
    I64ArrayAttr:$sizes,
    I64ArrayAttr:$strides
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $offsets `]` `[` $sizes `]` `[` $strides `]`
    attr-dict `:` type($source) `to` type($result)
  }];
}

// 2. Pad - 边界填充
def BatchComp_PadOp : BatchComp_Op<"pad", [Pure]> {
  let summary = "Pad tensor to required size";
  let arguments = (ins
    AnyRankedTensor:$source,
    I64ArrayAttr:$padding,  // [[left, right], [top, bottom]]
    F32Attr:$pad_value
  );
  let results = (outs AnyRankedTensor:$result);
}

// 3. TileMatmul - 固定大小矩阵乘法
def BatchComp_TileMatmulOp : BatchComp_Op<"tile_matmul", [Pure]> {
  let summary = "Fixed-size tile matrix multiplication";
  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    OptionalAttr<I64ArrayAttr>:$actual_size  // 实际有效大小
  );
  let results = (outs AnyRankedTensor:$result);

  let extraClassDeclaration = [{
    // 验证 tile 大小符合硬件要求
    LogicalResult verifyTileSize();
  }];
}

// 4. Accumulate - 累加操作（K维规约）
def BatchComp_AccumulateOp : BatchComp_Op<"accumulate", [Pure]> {
  let summary = "Accumulate tile results for K-dimension reduction";
  let arguments = (ins
    AnyRankedTensor:$accumulator,
    AnyRankedTensor:$update
  );
  let results = (outs AnyRankedTensor:$result);
}

//===----------------------------------------------------------------------===//
// 批组合操作
//===----------------------------------------------------------------------===//

// 5. TileInfo - 描述一个tile计算
def BatchComp_TileInfoOp : BatchComp_Op<"tile_info"> {
  let summary = "Describe a tile computation";
  let arguments = (ins
    I64ArrayAttr:$a_slice,    // [m_start, m_end, k_start, k_end]
    I64ArrayAttr:$b_slice,    // [k_start, k_end, n_start, n_end]
    I64ArrayAttr:$c_slice,    // [m_start, m_end, n_start, n_end]
    BoolAttr:$accumulate,     // 是否累加到C
    OptionalAttr<BoolAttr>:$is_nop  // 是否是填充NOP
  );
  let results = (outs TileInfoType:$tile_info);
}

// 6. Batch - 批次容器
def BatchComp_BatchOp : BatchComp_Op<"batch"> {
  let summary = "Create a batch of tiles";
  let arguments = (ins
    Variadic<TileInfoType>:$tiles,
    I64Attr:$batch_size,
    HardwareConfigAttr:$hw_config
  );
  let results = (outs BatchType:$batch);

  let verifier = [{
    // 验证批大小符合硬件要求
    return verifyBatchSize();
  }];
}

// 7. ExecuteBatch - 执行一个批次
def BatchComp_ExecuteBatchOp : BatchComp_Op<"execute_batch"> {
  let summary = "Execute a batch of tiles";
  let arguments = (ins
    BatchType:$batch,
    AnyRankedTensor:$input_a,
    AnyRankedTensor:$input_b,
    AnyRankedTensor:$output
  );
  let results = (outs AnyRankedTensor:$result);
}
```

### 2.3 完整的转换示例

```mlir
// 转换后的 MLIR - 100x200 @ 200x300 矩阵乘法
module @output_batched attributes {
  batch_comp.config = {
    hardware = "NPU",
    batch_size = 16,
    tile_size = [16, 16, 16],
    must_fill = true
  }
} {
  func.func @matmul_tiled(%A: tensor<100x200xf32>,
                          %B: tensor<200x300xf32>) -> tensor<100x300xf32> {
    // 初始化输出
    %c0 = arith.constant 0.0 : f32
    %C_init = linalg.fill ins(%c0) outs(tensor<100x300xf32>)

    // ===== 分解维度 =====
    // M: 100 = 6*16 + 4 (需要7个M tiles)
    // N: 300 = 18*16 + 12 (需要19个N tiles)
    // K: 200 = 12*16 + 8 (需要13个K tiles)
    // 总tiles: 7*19*13 = 1729个
    // 批次数: ceil(1729/16) = 109批

    // ===== Batch 0: 处理前16个tiles =====
    // Tile 0: C[0:16,0:16] = A[0:16,0:16] @ B[0:16,0:16]
    %a_0 = batch_comp.slice %A[0,0][16,16][1,1] :
           tensor<100x200xf32> to tensor<16x16xf32>
    %b_0 = batch_comp.slice %B[0,0][16,16][1,1] :
           tensor<200x300xf32> to tensor<16x16xf32>
    %c_0 = batch_comp.tile_matmul %a_0, %b_0 :
           (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>

    // Tile 1: C[0:16,0:16] += A[0:16,16:32] @ B[16:32,0:16] (K维累加)
    %a_1 = batch_comp.slice %A[0,16][16,16][1,1] :
           tensor<100x200xf32> to tensor<16x16xf32>
    %b_1 = batch_comp.slice %B[16,0][16,16][1,1] :
           tensor<200x300xf32> to tensor<16x16xf32>
    %c_1_tmp = batch_comp.tile_matmul %a_1, %b_1 :
               (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %c_1 = batch_comp.accumulate %c_0, %c_1_tmp :
           (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>

    // ... Tiles 2-12: 继续K维累加 ...

    // Tile 12: 最后的K维（需要padding）
    %a_12_raw = batch_comp.slice %A[0,192][16,8][1,1] :
                tensor<100x200xf32> to tensor<16x8xf32>
    %a_12 = batch_comp.pad %a_12_raw {padding=[[0,0],[0,8]], pad_value=0.0} :
            tensor<16x8xf32> to tensor<16x16xf32>
    %b_12_raw = batch_comp.slice %B[192,0][8,16][1,1] :
                tensor<200x300xf32> to tensor<8x16xf32>
    %b_12 = batch_comp.pad %b_12_raw {padding=[[0,8],[0,0]], pad_value=0.0} :
            tensor<8x16xf32> to tensor<16x16xf32>
    %c_12_tmp = batch_comp.tile_matmul %a_12, %b_12 :
                (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<16x16xf32>
    %c_final_0_0 = batch_comp.accumulate %c_11, %c_12_tmp

    // 将结果写回C[0:16,0:16]
    %C_0 = tensor.insert_slice %c_final_0_0 into %C_init[0,0][16,16][1,1]

    // ===== 批组合优化表示 =====
    // 创建tile信息
    %tile_info_0 = batch_comp.tile_info {
      a_slice = [0, 16, 0, 16],
      b_slice = [0, 16, 0, 16],
      c_slice = [0, 16, 0, 16],
      accumulate = false
    }

    %tile_info_1 = batch_comp.tile_info {
      a_slice = [0, 16, 16, 32],
      b_slice = [16, 32, 0, 16],
      c_slice = [0, 16, 0, 16],
      accumulate = true
    }

    // ... 创建所有1729个tile信息 ...

    // 批组合优化（调用CP-SAT）
    %batch_0 = batch_comp.batch %tile_info_0, %tile_info_1, ..., %nop_15
               {batch_size = 16, hw_config = #npu_config}

    %C_batch_0 = batch_comp.execute_batch %batch_0, %A, %B, %C_init

    // ... 执行剩余108批 ...

    return %C_final : tensor<100x300xf32>
  }
}
```

---

## 3. 批组合优化算法

### 3.1 通用优化框架

```python
class BatchCompositionOptimizer:
    """通用批组合优化器"""

    def optimize(self, tiles: List[TileInfo], hw_config: HardwareConfig) -> List[Batch]:
        """
        核心优化接口

        Args:
            tiles: 所有需要执行的tile操作
            hw_config: 硬件配置（batch_size, must_fill等）

        Returns:
            优化后的批次列表
        """

        # 根据硬件类型选择优化策略
        if hw_config.hardware == "NPU":
            # 固定16个一批，CP-SAT优化
            return self._optimize_fixed_batch(tiles, 16)
        elif hw_config.hardware == "GPU":
            # Warp级别调度，32线程
            return self._optimize_warp_batch(tiles, 32)
        elif hw_config.hardware == "TPU":
            # Systolic array调度
            return self._optimize_systolic_batch(tiles, 128)

    def _optimize_fixed_batch(self, tiles: List[TileInfo], batch_size: int):
        """NPU风格：固定大小批次，最小化NOP"""

        if len(tiles) <= 50:
            # 小规模：CP-SAT全局最优
            return self._cpsat_optimize(tiles, batch_size)
        elif len(tiles) <= 1000:
            # 中等规模：改进贪心
            return self._improved_greedy(tiles, batch_size)
        else:
            # 大规模：基础贪心
            return self._basic_greedy(tiles, batch_size)
```

### 3.2 CP-SAT 优化实现

```python
from ortools.sat.python import cp_model

def cpsat_optimize(tiles: List[TileInfo], batch_size: int = 16) -> List[Batch]:
    """使用 CP-SAT 求解器进行全局优化"""

    model = cp_model.CpModel()
    num_tiles = len(tiles)
    max_batches = (num_tiles + batch_size - 1) // batch_size

    # 决策变量：x[i,j] = tile i 是否在 batch j 中
    x = {}
    for i in range(num_tiles):
        for j in range(max_batches):
            x[i, j] = model.NewBoolVar(f'tile_{i}_batch_{j}')

    # 约束1：每个tile必须被分配
    for i in range(num_tiles):
        model.Add(sum(x[i, j] for j in range(max_batches)) == 1)

    # 约束2：每批正好batch_size个（包括NOP）
    batch_used = []
    for j in range(max_batches):
        used = model.NewBoolVar(f'batch_{j}_used')
        batch_used.append(used)

        # 如果批次被使用，必须有batch_size个元素
        tiles_in_batch = sum(x[i, j] for i in range(num_tiles))
        model.Add(tiles_in_batch <= batch_size)
        model.Add(tiles_in_batch > 0).OnlyEnforceIf(used)
        model.Add(tiles_in_batch == 0).OnlyEnforceIf(used.Not())

    # 约束3：相关tiles尽量在同一批（K维累加）
    for i in range(num_tiles):
        tile = tiles[i]
        if tile.accumulate:  # K维累加的tile
            # 找到对应的前一个K tile
            prev_tile_idx = find_previous_k_tile(tiles, i)
            if prev_tile_idx >= 0:
                # 软约束：尽量在同一批
                for j in range(max_batches):
                    same_batch = model.NewBoolVar(f'same_batch_{i}_{prev_tile_idx}_{j}')
                    model.Add(x[i, j] + x[prev_tile_idx, j] == 2).OnlyEnforceIf(same_batch)

    # 目标：最小化批次数和NOP数
    num_batches = sum(batch_used)
    num_nops = sum(
        (batch_size - sum(x[i, j] for i in range(num_tiles))) * batch_used[j]
        for j in range(max_batches)
    )

    # 多目标优化：1000*批次数 + NOP数
    model.Minimize(1000 * num_batches + num_nops)

    # 求解
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10  # 限制求解时间
    status = solver.Solve(model)

    # 提取结果
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return extract_batches(solver, x, tiles, batch_size)
    else:
        # 降级到贪心算法
        return greedy_fallback(tiles, batch_size)
```

### 3.3 硬件适配层

```python
class HardwareAdapter:
    """硬件适配层，处理不同硬件的特殊需求"""

    @staticmethod
    def create_config(hardware: str) -> dict:
        configs = {
            "NPU": {
                "batch_size": 16,
                "must_fill": True,
                "tile_size": [16, 16, 16],
                "supports_variable_tile": True,
                "padding_type": "nop"
            },
            "GPU": {
                "batch_size": 32,  # Warp size
                "must_fill": False,
                "tile_size": [16, 16, 16],  # Tensor Core
                "supports_variable_tile": False,
                "padding_type": "predicate"
            },
            "TPU": {
                "batch_size": 128,
                "must_fill": False,
                "tile_size": [128, 128, 128],
                "supports_variable_tile": False,
                "padding_type": "zero"
            }
        }
        return configs.get(hardware, configs["NPU"])
```

---

## 4. 实施计划

### 4.1 第一阶段：核心框架（Month 1）

- [ ] 实现 Batch Composition Dialect
- [ ] 基础 Pass：TOSA → BatchComp
- [ ] NPU 配置和贪心调度
- [ ] 单元测试框架

### 4.2 第二阶段：优化算法（Month 2）

- [ ] CP-SAT 集成
- [ ] 多目标优化
- [ ] K维累加优化
- [ ] 性能基准测试

### 4.3 第三阶段：多硬件支持（Month 3）

- [ ] GPU Tensor Core 适配
- [ ] TPU 模拟支持
- [ ] 硬件配置系统
- [ ] 跨平台测试

### 4.4 第四阶段：生产化（Month 4-6）

- [ ] XLA Backend 集成
- [ ] PJRT Plugin 实现
- [ ] 性能调优
- [ ] 文档和示例

---

## 5. 关键创新点

### 5.1 统一抽象
- **问题本质**：识别批组合优化是共同问题
- **通用框架**：一个优化器支持多种硬件
- **算法复用**：CP-SAT等算法跨硬件共享

### 5.2 实用价值
- **降低复杂度**：不需要为每个硬件写独立backend
- **提高性能**：全局优化优于局部贪心
- **易于扩展**：新硬件只需配置文件

### 5.3 学术贡献
- **新的问题定义**：批组合优化作为独立问题
- **跨硬件优化**：统一的调度框架
- **实验验证**：多硬件性能对比

---

## 6. 性能目标

| 指标 | NPU | GPU | TPU |
|------|-----|-----|-----|
| **硬件利用率** | >85% | >90% | >95% |
| **调度开销** | <100ms | <50ms | <200ms |
| **内存效率** | >90% | >85% | >90% |
| **扩展性** | 10K tiles | 100K tiles | 1M tiles |

---

## 7. 风险和缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 硬件差异过大 | 高 | 提供硬件特定扩展点 |
| CP-SAT扩展性 | 中 | 分层调度，大问题分解 |
| 性能回归 | 中 | 保留硬件原生路径 |
| 集成复杂度 | 低 | 渐进式集成策略 |

---

## 8. 结论

Batch Composition Optimization 框架通过识别 tile-based 加速器的共同优化模式，提供了一个优雅的统一解决方案。这不仅简化了编译器开发，还通过全局优化提高了性能。该设计的核心价值在于：

1. **简化抽象**：批组合是正确的抽象层次
2. **算法复用**：CP-SAT 等优化跨硬件共享
3. **实用价值**：可直接集成到 XLA/MLIR 生态

这个框架将显著降低新硬件的集成成本，同时提供更好的性能。

---

**作者**: NPU Compiler Team
**版本**: 2.0
**状态**: 设计完成，待实施