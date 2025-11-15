# MLIR Pass 开发教程

**完整的BatchComp MLIR Pass开发指南**

本教程将带您从零开始学习如何使用和开发BatchComp MLIR Passes，涵盖从基础概念到高级优化。

---

## 📋 目录

1. [MLIR基础概念](#1-mlir基础概念)
2. [BatchComp Pass管道概览](#2-batchcomp-pass管道概览)
3. [第一个Pass使用示例](#3-第一个pass使用示例)
4. [深入理解IR转换](#4-深入理解ir转换)
5. [自定义Pass开发](#5-自定义pass开发)
6. [测试和调试](#6-测试和调试)
7. [Pass集成到编译流程](#7-pass集成到编译流程)
8. [性能优化技巧](#8-性能优化技巧)
9. [最佳实践](#9-最佳实践)

---

## 1. MLIR基础概念

### 1.1 什么是MLIR？

MLIR (Multi-Level Intermediate Representation) 是一个可扩展的编译器基础设施，允许您：
- 定义自己的IR方言（Dialect）
- 编写编译器转换Pass
- 在不同抽象层次间转换代码

### 1.2 核心概念

```mlir
// MLIR代码示例
func.func @example(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
  // Operation (操作)
  %result = tosa.matmul %arg0, %arg0 : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>

  // Return
  return %result : tensor<64x64xf32>
}
```

**关键术语：**
- **Operation (操作)**: MLIR的基本单元，如 `tosa.matmul`
- **Dialect (方言)**: 相关操作的集合，如 `tosa`, `batch_comp`
- **Type (类型)**: 值的类型，如 `tensor<64x64xf32>`
- **Attribute (属性)**: 编译时常量，如 `tile_size = [16, 16, 16]`
- **Pass (转换)**: 修改IR的编译器转换

### 1.3 BatchComp Dialect概览

BatchComp提供两层IR：

**高层IR (用户友好):**
```mlir
// 生成tiles
%tiles = batch_comp.generate_tiles %A, %B {
  tile_size = [16, 16, 16]
} : (tensor<100x200xf32>, tensor<200x300xf32>) -> !batch_comp.tileset

// 调度batches
%schedule = batch_comp.schedule_batches %tiles {
  hardware = "npu",
  batch_size = 16 : i32,
  algorithm = "cpsat"
} : !batch_comp.tileset -> !batch_comp.schedule

// 执行
%result = batch_comp.execute_schedule %schedule {
  hardware = "npu"
} : !batch_comp.schedule -> tensor<100x300xf32>
```

**低层IR (精确控制):**
```mlir
// 创建batch
%batch = batch_comp.create_batch {
  batch_id = 0 : i32,
  size = 16 : i32
} : !batch_comp.batch

// 执行batch
batch_comp.execute_batch %batch {hardware = "npu"}

// 累加结果
%result = batch_comp.accumulate %partial_result, %current_result
```

---

## 2. BatchComp Pass管道概览

### 2.1 完整Pass流程

```
TOSA Dialect
    ↓
BatchCompTileScheduler Pass
    ↓ (生成高层BatchComp IR)
BatchCompLowering Pass
    ↓ (降低到低层BatchComp IR)
BatchCompToNPU Pass
    ↓ (转换到NPU Dialect)
NPU Dialect
```

### 2.2 各Pass职责

| Pass | 输入 | 输出 | 功能 |
|------|------|------|------|
| **BatchCompTileScheduler** | TOSA matmul | 高层BatchComp IR | 调用Python优化器，生成批处理计划 |
| **BatchCompLowering** | 高层BatchComp IR | 低层BatchComp IR | 展开高层操作为详细控制流 |
| **BatchCompToNPU** | 低层BatchComp IR | NPU Dialect | 映射到硬件特定操作 |
| **BatchCompCanonicalize** | BatchComp IR | 优化的BatchComp IR | 常规化和优化 |

### 2.3 数据流

```
输入: tosa.matmul %A, %B

TileScheduler Pass:
  1. 提取矩阵维度: M=100, K=200, N=300
  2. 调用Python优化器: optimize_tiles(tiles, "cpsat")
  3. 生成3个高层操作: generate_tiles → schedule_batches → execute_schedule

Lowering Pass:
  1. generate_tiles → slice操作生成实际tiles
  2. execute_schedule → 循环遍历batches，调用execute_batch + accumulate

ToNPU Pass:
  1. execute_batch → npu.tc_batch (Tensor Core batch操作)
  2. accumulate → npu.accumulate

输出: npu.tc_batch + npu.accumulate
```

---

## 3. 第一个Pass使用示例

### 3.1 准备MLIR输入文件

创建 `example.mlir`:

```mlir
module {
  func.func @simple_matmul(
    %A: tensor<64x128xf32>,
    %B: tensor<128x256xf32>
  ) -> tensor<64x256xf32> {
    %C = tosa.matmul %A, %B :
      (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
    return %C : tensor<64x256xf32>
  }
}
```

### 3.2 运行TileScheduler Pass

```bash
# 基本用法
mlir-opt example.mlir \
  --batch-comp-tile-scheduler="hardware=npu tile-size=16,16,16 algorithm=cpsat"

# 启用多矩阵优化
mlir-opt example.mlir \
  --batch-comp-tile-scheduler="hardware=npu tile-size=16,16,16 algorithm=cpsat enable-multi-matmul=true"

# 指定自定义硬件配置
mlir-opt example.mlir \
  --batch-comp-tile-scheduler="hardware=custom config-file=my_hw.yaml"
```

### 3.3 理解输出

**转换前 (TOSA):**
```mlir
%C = tosa.matmul %A, %B : (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
```

**转换后 (BatchComp高层IR):**
```mlir
// 步骤1: 生成tiles
%tiles = batch_comp.generate_tiles %A, %B {
  tile_size = [16, 16, 16],
  matrix_id = 0 : i32
} : (tensor<64x128xf32>, tensor<128x256xf32>) -> !batch_comp.tileset

// 步骤2: 优化调度
%schedule = batch_comp.schedule_batches %tiles {
  hardware = "npu",
  batch_size = 16 : i32,
  algorithm = "cpsat"
} : !batch_comp.tileset -> !batch_comp.schedule

// 步骤3: 执行
%C = batch_comp.execute_schedule %schedule {
  hardware = "npu"
} : !batch_comp.schedule -> tensor<64x256xf32>
```

---

## 4. 深入理解IR转换

### 4.1 TileScheduler Pass详解

**核心逻辑 (简化):**

```cpp
// lib/BatchComp/BatchCompTileScheduler.cpp

void runOnOperation() override {
  // 1. 收集所有matmul操作
  SmallVector<MatmulInfo> matmuls = collectMatmuls(getOperation());

  // 2. 为每个matmul生成tiles
  for (auto &matmul : matmuls) {
    TileInfo tiles = generateTiles(matmul, tileSize_);
  }

  // 3. 调用Python优化器
  BatchScheduleResult schedule = pythonOptimizer_.optimize(tiles, algorithm_);

  // 4. 生成BatchComp IR
  for (auto &matmul : matmuls) {
    generateBatchCompIR(matmul, schedule);
  }
}
```

**generateBatchCompIR函数:**

```cpp
void generateBatchCompIR(MatmulInfo &matmul, BatchScheduleResult &schedule) {
  OpBuilder builder(matmul.op);
  Location loc = matmul.op.getLoc();

  // 创建generate_tiles操作
  auto generateTilesOp = builder.create<batch_comp::GenerateTilesOp>(
    loc,
    batch_comp::TilesetType::get(context),
    matmul.lhs,  // %A
    matmul.rhs,  // %B
    tileSizeAttr,
    matrixIdAttr
  );

  // 创建schedule_batches操作
  auto scheduleBatchesOp = builder.create<batch_comp::ScheduleBatchesOp>(
    loc,
    batch_comp::ScheduleType::get(context),
    generateTilesOp.getResult(),
    hardwareAttr,
    batchSizeAttr,
    algorithmAttr
  );

  // 创建execute_schedule操作
  auto executeScheduleOp = builder.create<batch_comp::ExecuteScheduleOp>(
    loc,
    matmul.op.getResult().getType(),
    scheduleBatchesOp.getResult(),
    hardwareAttr
  );

  // 替换原始matmul
  matmul.op.getResult().replaceAllUsesWith(executeScheduleOp.getResult());
  matmul.op.erase();
}
```

### 4.2 Lowering Pass详解

**高层 → 低层转换:**

```mlir
// 输入: 高层IR
%result = batch_comp.execute_schedule %schedule {hardware = "npu"}

// 输出: 低层IR
%init = arith.constant dense<0.0> : tensor<64x256xf32>
%final = scf.for %batch_id = %c0 to %num_batches step %c1 iter_args(%acc = %init) {
  // 执行batch
  %batch_result = batch_comp.execute_batch %schedule, %batch_id {hardware = "npu"}

  // 累加结果
  %new_acc = batch_comp.accumulate %acc, %batch_result
  scf.yield %new_acc
}
```

**实现模式:**

```cpp
// lib/BatchComp/BatchCompLowering.cpp

class ExecuteScheduleLowering : public OpRewritePattern<ExecuteScheduleOp> {
  LogicalResult matchAndRewrite(ExecuteScheduleOp op, PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    // 获取batch数量（从schedule中）
    int numBatches = getNumBatches(op.getSchedule());

    // 创建初始累加器（零矩阵）
    Value init = createZeroTensor(rewriter, loc, op.getType());

    // 创建循环
    auto forOp = rewriter.create<scf::ForOp>(
      loc,
      /*lowerBound=*/rewriter.create<arith::ConstantIndexOp>(loc, 0),
      /*upperBound=*/rewriter.create<arith::ConstantIndexOp>(loc, numBatches),
      /*step=*/rewriter.create<arith::ConstantIndexOp>(loc, 1),
      /*iterArgs=*/init
    );

    // 循环体
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(forOp.getBody());

    // 执行batch
    Value batchResult = rewriter.create<ExecuteBatchOp>(
      loc, op.getType(), op.getSchedule(), forOp.getInductionVar()
    );

    // 累加
    Value newAcc = rewriter.create<AccumulateOp>(
      loc, forOp.getRegionIterArg(0), batchResult
    );

    rewriter.create<scf::YieldOp>(loc, newAcc);

    // 替换原操作
    rewriter.replaceOp(op, forOp.getResult(0));
    return success();
  }
};
```

### 4.3 ToNPU Pass详解

**BatchComp → NPU映射:**

```mlir
// 输入: BatchComp低层IR
%result = batch_comp.execute_batch %schedule, %batch_id {hardware = "npu"}

// 输出: NPU Dialect
%result = npu.tc_batch %tiles_A, %tiles_B, %batch_id {
  tile_size = [16, 16, 16],
  batch_size = 16 : i32
}
```

**实现:**

```cpp
// lib/BatchComp/BatchCompToNPU.cpp

class ExecuteBatchToNPU : public OpRewritePattern<ExecuteBatchOp> {
  LogicalResult matchAndRewrite(ExecuteBatchOp op, PatternRewriter &rewriter) const override {
    // 提取batch信息
    auto schedule = op.getSchedule();
    auto batchId = op.getBatchId();

    // 获取该batch的tile信息
    SmallVector<Value> tilesA, tilesB;
    extractTilesForBatch(schedule, batchId, tilesA, tilesB);

    // 创建NPU tensor core batch操作
    auto npuOp = rewriter.create<npu::TensorCoreBatchOp>(
      op.getLoc(),
      op.getType(),
      tilesA,
      tilesB,
      batchId,
      op->getAttr("hardware")
    );

    rewriter.replaceOp(op, npuOp.getResult());
    return success();
  }
};
```

---

## 5. 自定义Pass开发

### 5.1 创建新Pass

假设我们要创建一个自动选择最优tile size的Pass。

**步骤1: 定义Pass类**

```cpp
// lib/BatchComp/BatchCompAutoTiler.cpp

#include "mlir/Pass/Pass.h"
#include "BatchComp/BatchCompDialect.h"
#include "BatchComp/BatchCompOps.h"

namespace mlir {
namespace batch_comp {

class BatchCompAutoTilerPass
    : public PassWrapper<BatchCompAutoTilerPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(BatchCompAutoTilerPass)

  StringRef getArgument() const override { return "batch-comp-auto-tiler"; }
  StringRef getDescription() const override {
    return "Automatically determine optimal tile sizes";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // 遍历所有函数
    module.walk([&](func::FuncOp func) {
      // 分析函数中的matmul操作
      SmallVector<MatmulInfo> matmuls;
      collectMatmuls(func, matmuls);

      // 为每个matmul确定最优tile size
      for (auto &matmul : matmuls) {
        auto optimalSize = determineOptimalTileSize(matmul);
        annotateTileSize(matmul.op, optimalSize);
      }
    });
  }

private:
  std::array<int64_t, 3> determineOptimalTileSize(const MatmulInfo &matmul) {
    // 获取矩阵维度
    int64_t M = matmul.M;
    int64_t K = matmul.K;
    int64_t N = matmul.N;

    // 简单启发式: 选择能整除且接近硬件限制的size
    int64_t tileM = findOptimalDivisor(M, 16, 32);
    int64_t tileK = findOptimalDivisor(K, 16, 32);
    int64_t tileN = findOptimalDivisor(N, 16, 32);

    return {tileM, tileK, tileN};
  }

  int64_t findOptimalDivisor(int64_t dim, int64_t minSize, int64_t maxSize) {
    // 从maxSize向下找第一个能整除dim的大小
    for (int64_t size = maxSize; size >= minSize; size--) {
      if (dim % size == 0) return size;
    }
    return minSize;  // 默认最小size
  }

  void annotateTileSize(Operation *op, std::array<int64_t, 3> size) {
    // 添加tile_size属性
    Builder builder(op->getContext());
    op->setAttr("tile_size", builder.getI64ArrayAttr(size));
  }
};

} // namespace batch_comp
} // namespace mlir
```

**步骤2: 注册Pass**

```cpp
// include/BatchComp/BatchCompPasses.h

std::unique_ptr<Pass> createBatchCompAutoTilerPass();

// lib/BatchComp/BatchCompPasses.cpp

std::unique_ptr<Pass> createBatchCompAutoTilerPass() {
  return std::make_unique<BatchCompAutoTilerPass>();
}

void registerBatchCompPasses() {
  PassRegistration<BatchCompAutoTilerPass>();
}
```

**步骤3: 使用新Pass**

```bash
mlir-opt example.mlir \
  --batch-comp-auto-tiler \
  --batch-comp-tile-scheduler="hardware=npu algorithm=cpsat"
```

### 5.2 编写Pass测试

```mlir
// test/BatchComp/auto-tiler.mlir

// RUN: mlir-opt %s --batch-comp-auto-tiler | FileCheck %s

module {
  func.func @test_auto_tiler(%A: tensor<64x128xf32>, %B: tensor<128x256xf32>) -> tensor<64x256xf32> {
    // CHECK: tosa.matmul
    // CHECK-SAME: tile_size = [32, 32, 32]
    %C = tosa.matmul %A, %B : (tensor<64x128xf32>, tensor<128x256xf32>) -> tensor<64x256xf32>
    return %C : tensor<64x256xf32>
  }
}
```

---

## 6. 测试和调试

### 6.1 单元测试

**编写lit测试:**

```mlir
// test/BatchComp/tile-scheduler.mlir

// RUN: mlir-opt %s --batch-comp-tile-scheduler="hardware=npu tile-size=16,16,16" | FileCheck %s

module {
  // CHECK-LABEL: func.func @simple_matmul
  func.func @simple_matmul(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK-NOT: tosa.matmul
    // CHECK: batch_comp.generate_tiles
    // CHECK-SAME: tile_size = [16, 16, 16]
    // CHECK: batch_comp.schedule_batches
    // CHECK-SAME: hardware = "npu"
    // CHECK-SAME: algorithm = "cpsat"
    // CHECK: batch_comp.execute_schedule
    %C = tosa.matmul %A, %B : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %C : tensor<64x64xf32>
  }
}
```

**运行测试:**

```bash
# 运行所有BatchComp测试
lit test/BatchComp/

# 运行单个测试
lit test/BatchComp/tile-scheduler.mlir

# 显示详细输出
lit -v test/BatchComp/tile-scheduler.mlir
```

### 6.2 调试技巧

**1. 打印IR在各Pass之间的变化:**

```bash
mlir-opt example.mlir \
  --print-ir-before-all \
  --print-ir-after-all \
  --batch-comp-tile-scheduler="hardware=npu"
```

**2. 只打印特定Pass的IR:**

```bash
mlir-opt example.mlir \
  --print-ir-after=batch-comp-tile-scheduler \
  --batch-comp-tile-scheduler="hardware=npu"
```

**3. 使用mlir-opt的调试选项:**

```bash
# 打印Pass管道
mlir-opt example.mlir --print-op-graph

# 验证IR有效性
mlir-opt example.mlir --verify-each=true

# 显示Pass统计
mlir-opt example.mlir --mlir-pass-statistics
```

**4. 在代码中添加调试输出:**

```cpp
void runOnOperation() override {
  llvm::errs() << "Running BatchCompTileScheduler on: " << getOperation() << "\n";

  // 打印matmul信息
  for (auto &matmul : matmuls) {
    llvm::errs() << "Found matmul: M=" << matmul.M
                 << " K=" << matmul.K
                 << " N=" << matmul.N << "\n";
  }
}
```

**5. 使用LLVM_DEBUG:**

```cpp
#define DEBUG_TYPE "batch-comp-tile-scheduler"

LLVM_DEBUG(llvm::dbgs() << "Optimizing with algorithm: " << algorithm_ << "\n");
```

运行时启用调试输出:

```bash
mlir-opt example.mlir \
  --debug-only=batch-comp-tile-scheduler \
  --batch-comp-tile-scheduler="hardware=npu"
```

### 6.3 性能分析

**1. 测量Pass执行时间:**

```bash
mlir-opt example.mlir \
  --mlir-timing \
  --batch-comp-tile-scheduler="hardware=npu"
```

**2. 生成性能报告:**

```bash
mlir-opt example.mlir \
  --mlir-timing \
  --mlir-timing-display=list \
  --batch-comp-tile-scheduler="hardware=npu" \
  2>&1 | grep "batch-comp"
```

---

## 7. Pass集成到编译流程

### 7.1 定义Pass管道

**创建完整的优化管道:**

```cpp
// tools/batch-comp-opt/batch-comp-opt.cpp

void buildBatchCompPipeline(OpPassManager &pm) {
  // 前端优化
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());

  // BatchComp核心Passes
  pm.addPass(createBatchCompAutoTilerPass());
  pm.addPass(createBatchCompTileSchedulerPass());
  pm.addPass(createBatchCompLoweringPass());
  pm.addPass(createBatchCompCanonicalizePass());
  pm.addPass(createBatchCompToNPUPass());

  // 后端优化
  pm.addPass(createCSEPass());
}
```

**使用管道:**

```bash
mlir-opt example.mlir --pass-pipeline="builtin.module(func.func(cse,canonicalize),batch-comp-tile-scheduler,batch-comp-lowering,batch-comp-to-npu)"
```

### 7.2 条件Pass执行

```cpp
void buildConditionalPipeline(OpPassManager &pm, bool enableMultiMatmul) {
  pm.addPass(createBatchCompTileSchedulerPass(enableMultiMatmul));

  if (enableMultiMatmul) {
    // 启用跨矩阵优化时添加额外Pass
    pm.addPass(createBatchCompMultiMatmulFusionPass());
  }

  pm.addPass(createBatchCompLoweringPass());
}
```

### 7.3 嵌套Pass管理

```cpp
// 只在特定函数上运行Passes
void buildNestedPipeline(OpPassManager &pm) {
  // 模块级Pass
  pm.addPass(createSymbolDCEPass());

  // 函数级Pass
  OpPassManager &funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(createBatchCompTileSchedulerPass());
  funcPM.addPass(createBatchCompLoweringPass());

  // 回到模块级
  pm.addPass(createInlinerPass());
}
```

---

## 8. 性能优化技巧

### 8.1 避免不必要的IR遍历

**❌ 低效:**

```cpp
void runOnOperation() override {
  // 多次遍历整个模块
  getOperation().walk([](tosa::MatmulOp op) { /* ... */ });
  getOperation().walk([](func::FuncOp op) { /* ... */ });
  getOperation().walk([](tosa::MatmulOp op) { /* ... */ });
}
```

**✅ 高效:**

```cpp
void runOnOperation() override {
  // 单次遍历收集所有需要的信息
  SmallVector<tosa::MatmulOp> matmuls;
  SmallVector<func::FuncOp> funcs;

  getOperation().walk([&](Operation *op) {
    if (auto matmul = dyn_cast<tosa::MatmulOp>(op))
      matmuls.push_back(matmul);
    else if (auto func = dyn_cast<func::FuncOp>(op))
      funcs.push_back(func);
  });

  // 处理收集的操作
  processMatmuls(matmuls);
  processFuncs(funcs);
}
```

### 8.2 使用PatternRewriter高效

**❌ 低效:**

```cpp
LogicalResult matchAndRewrite(GenerateTilesOp op, PatternRewriter &rewriter) const {
  // 创建多个临时值
  auto temp1 = rewriter.create<...>();
  auto temp2 = rewriter.create<...>();
  // ...

  // 最后才替换
  rewriter.replaceOp(op, finalResult);
}
```

**✅ 高效:**

```cpp
LogicalResult matchAndRewrite(GenerateTilesOp op, PatternRewriter &rewriter) const {
  // 使用 replaceOpWithNewOp 一步完成
  rewriter.replaceOpWithNewOp<NewOp>(op, args...);
  return success();
}
```

### 8.3 缓存分析结果

```cpp
class BatchCompTileSchedulerPass : public ... {
  // 缓存Python优化器结果
  DenseMap<Operation*, BatchScheduleResult> scheduleCache_;

  BatchScheduleResult getOrComputeSchedule(Operation *op) {
    auto it = scheduleCache_.find(op);
    if (it != scheduleCache_.end())
      return it->second;

    auto result = pythonOptimizer_.optimize(...);
    scheduleCache_[op] = result;
    return result;
  }
};
```

---

## 9. 最佳实践

### 9.1 Pass设计原则

1. **单一职责**: 每个Pass只做一件事
   - ✅ BatchCompTileScheduler: 只负责调度
   - ✅ BatchCompLowering: 只负责降低
   - ❌ 不要在一个Pass中混合多种转换

2. **幂等性**: Pass多次运行应该产生相同结果
   ```cpp
   // ✅ 好: 检查操作是否已转换
   if (op->hasAttr("already_scheduled"))
     return;

   // ❌ 坏: 盲目转换，可能重复处理
   transformOperation(op);
   ```

3. **保持IR有效性**: 转换后IR必须通过验证
   ```cpp
   void runOnOperation() override {
     // 执行转换...

     // 验证结果
     if (failed(verify()))
       signalPassFailure();
   }
   ```

### 9.2 错误处理

```cpp
LogicalResult matchAndRewrite(Op op, PatternRewriter &rewriter) const {
  // 检查前置条件
  if (!isValidForTransform(op))
    return failure();

  // 执行转换
  if (failed(performTransform(op, rewriter))) {
    // 报告详细错误
    op.emitError("Failed to transform: ") << getErrorMessage();
    return failure();
  }

  return success();
}
```

### 9.3 文档和测试

**1. 为Pass编写文档:**

```cpp
/// This pass transforms TOSA matmul operations into BatchComp IR
/// by calling the Python optimizer to determine optimal batch schedules.
///
/// Input:
///   %C = tosa.matmul %A, %B
///
/// Output:
///   %tiles = batch_comp.generate_tiles %A, %B {...}
///   %schedule = batch_comp.schedule_batches %tiles {...}
///   %C = batch_comp.execute_schedule %schedule {...}
///
/// Options:
///   - hardware: Target hardware (npu, gpu, tpu)
///   - tile-size: Tile dimensions [M, K, N]
///   - algorithm: Optimization algorithm (greedy, improved_greedy, cpsat)
class BatchCompTileSchedulerPass : public ... {
```

**2. 为每个转换编写测试:**

```mlir
// 测试基本功能
// RUN: mlir-opt %s --batch-comp-tile-scheduler | FileCheck %s --check-prefix=BASIC

// 测试多矩阵优化
// RUN: mlir-opt %s --batch-comp-tile-scheduler="enable-multi-matmul=true" | FileCheck %s --check-prefix=MULTI

// 测试错误处理
// RUN: not mlir-opt %s --batch-comp-tile-scheduler="tile-size=0,0,0" 2>&1 | FileCheck %s --check-prefix=ERROR
```

### 9.4 代码组织

**推荐目录结构:**

```
lib/BatchComp/
├── BatchCompDialect.cpp          # Dialect定义
├── BatchCompOps.cpp              # 操作实现
├── BatchCompTypes.cpp            # 类型实现
├── Passes/
│   ├── BatchCompTileScheduler.cpp    # 主调度Pass
│   ├── BatchCompLowering.cpp         # 降低Pass
│   ├── BatchCompToNPU.cpp            # 后端转换
│   ├── BatchCompCanonicalize.cpp     # 优化Pass
│   └── PassDetail.h                  # Pass共享声明
├── Utils/
│   ├── PythonIntegration.cpp         # Python集成
│   └── TileUtils.cpp                 # Tile工具函数
└── CMakeLists.txt
```

---

## 10. 总结

### 10.1 学习路径回顾

您现在应该能够：

- ✅ 理解MLIR基础概念（Operation, Dialect, Pass）
- ✅ 理解BatchComp Pass管道的各个阶段
- ✅ 使用mlir-opt运行BatchComp Passes
- ✅ 阅读和理解IR转换
- ✅ 开发自定义Pass
- ✅ 编写测试和调试Pass
- ✅ 集成Pass到编译流程
- ✅ 应用性能优化技巧

### 10.2 下一步

1. **深入学习MLIR:**
   - 阅读 [MLIR官方文档](https://mlir.llvm.org/)
   - 学习 [Dialect定义](https://mlir.llvm.org/docs/DefiningDialects/)
   - 研究 [Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)

2. **扩展BatchComp:**
   - 添加对新硬件的支持
   - 实现更多优化Pass
   - 集成到实际编译器工具链

3. **性能调优:**
   - 分析编译时间
   - 优化生成代码质量
   - Benchmark不同优化策略

### 10.3 参考资源

**官方文档:**
- [MLIR Language Reference](https://mlir.llvm.org/docs/LangRef/)
- [MLIR Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)
- [MLIR Tutorials](https://mlir.llvm.org/docs/Tutorials/)

**BatchComp文档:**
- [Dialect设计](../design/BATCHCOMP_DIALECT_DESIGN.md)
- [Python优化器教程](PYTHON_OPTIMIZER_TUTORIAL.md)
- [API参考](../api/CPP_API.md)

**示例代码:**
- [MLIR示例](../../examples/mlir/)
- [Pass实现](../../lib/BatchComp/)

---

**💡 提示**: MLIR是一个强大但复杂的系统。不要试图一次学习所有内容，而是从实际问题出发，逐步深入理解各个概念。

**🎯 实践建议**: 从运行现有示例开始，然后逐步修改和扩展，最后尝试实现自己的Pass。
