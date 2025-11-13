# MLIR NPU Tile调度Pass设计方案

> 为固定tile数量的硬件加速器（Tensor Core / NPU Core）设计的自动tile调度Pass

---

## 1. 问题定义

### 硬件约束

```
自定义NPU Core:
- 固定并行度：16个tile
- 固定tile大小：[16, 16]
- 必须同时喂够16个tile才能启动
- 支持predication（可以mask无效tile）
```

### 输入场景

```mlir
// 场景1：单个不规则矩阵
func.func @matmul(%A: tensor<97x223xf32>, %B: tensor<223x137xf32>) {
  %C = linalg.matmul ins(%A, %B : ...) -> tensor<97x137xf32>
  return %C
}

// 场景2：多个小矩阵（Batching）
func.func @matmul_batch(
  %A0: tensor<33x48xf32>,
  %A1: tensor<50x30xf32>,
  %A2: tensor<20x25xf32>
) {
  %C0 = linalg.matmul ins(%A0, ...) -> tensor<33x48xf32>
  %C1 = linalg.matmul ins(%A1, ...) -> tensor<50x30xf32>
  %C2 = linalg.matmul ins(%A2, ...) -> tensor<20x25xf32>
  return %C0, %C1, %C2
}
```

### 目标

自动生成：
1. **Tile提取**：从输入矩阵提取16个tile
2. **Tile组合**：多矩阵的tile智能组合
3. **Padding处理**：不足16个时填充
4. **NPU调用**：生成NPU intrinsic调用

---

## 2. Pass设计概览

### 2.1 Pass定义（TableGen）

```tablegen
def NPUTileSchedulerPass : Pass<"npu-tile-scheduler", "func::FuncOp"> {
  let summary = "Schedule tiles for NPU core with fixed parallelism";

  let description = [{
    This pass schedules matrix tiles for hardware accelerators that require
    a fixed number of tiles (e.g., 16) to be processed in parallel.

    The pass:
    1. Analyzes input matrices and their dimensions
    2. Extracts tiles of fixed size (e.g., 16×16)
    3. Intelligently combines tiles from multiple matrices
    4. Minimizes padding overhead
    5. Generates NPU-specific intrinsic calls
  }];

  let options = [
    Option<"numTiles_", "num-tiles", "int64_t", /*default=*/"16",
           "Number of parallel tiles required by NPU">,
    Option<"tileK_", "tile-k", "int64_t", /*default=*/"16",
           "K dimension of each tile">,
    Option<"tileN_", "tile-n", "int64_t", /*default=*/"16",
           "N dimension of each tile">,
    Option<"strategy_", "strategy", "std::string", /*default=*/"\"optimized\"",
           "Scheduling strategy: simple, batching, or optimized">
  ];

  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect",
    "tensor::TensorDialect",
    "npu::NPUDialect"  // 自定义的NPU Dialect
  ];

  let constructor = "createNPUTileSchedulerPass()";
}
```

### 2.2 NPU Dialect定义

```tablegen
// npu_ops.td

def NPU_Dialect : Dialect {
  let name = "npu";
  let summary = "Custom NPU operations for tile-based computation";
}

// Tile batch容器
def NPU_TileBatchOp : NPU_Op<"create_batch"> {
  let summary = "Create a batch of tiles for NPU execution";
  let arguments = (ins I64Attr:$num_tiles);
  let results = (outs NPU_TileBatch:$batch);

  let assemblyFormat = [{
    `num_tiles` `=` $num_tiles attr-dict `:` type($batch)
  }];
}

// 提取tile
def NPU_ExtractTileOp : NPU_Op<"extract_tile"> {
  let summary = "Extract a tile from a tensor";
  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    DenseI64ArrayAttr:$tile_shape
  );
  let results = (outs AnyRankedTensor:$tile);

  let assemblyFormat = [{
    $source `[` $offsets `]` $tile_shape attr-dict
    `:` type($source) `->` type($tile)
  }];
}

// 添加tile到batch
def NPU_AddToBatchOp : NPU_Op<"add_to_batch"> {
  let summary = "Add a tile to the batch at specified slot";
  let arguments = (ins
    NPU_TileBatch:$batch,
    AnyRankedTensor:$tile,
    I64Attr:$slot,
    BoolAttr:$is_padding
  );
}

// 执行NPU计算
def NPU_ExecuteBatchOp : NPU_Op<"execute_batch"> {
  let summary = "Execute the batch on NPU core";
  let arguments = (ins
    NPU_TileBatch:$batch,
    StrAttr:$operation  // "matmul", "conv", etc.
  );
  let results = (outs NPU_ResultBatch:$results);
}

// 从结果中提取
def NPU_ExtractResultOp : NPU_Op<"extract_result"> {
  let summary = "Extract result from NPU execution";
  let arguments = (ins
    NPU_ResultBatch:$results,
    I64Attr:$slot
  );
  let results = (outs AnyRankedTensor:$result);
}
```

---

## 3. Pass实现流程

### 3.1 分析阶段（Analysis）

```cpp
class NPUTileSchedulerPass : public impl::NPUTileSchedulerPassBase<...> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // 步骤1：收集所有matmul操作
    SmallVector<linalg::MatmulOp> matmuls;
    funcOp.walk([&](linalg::MatmulOp matmul) {
      matmuls.push_back(matmul);
    });

    // 步骤2：分析每个matmul的维度
    SmallVector<MatrixInfo> matrices;
    for (auto matmul : matmuls) {
      MatrixInfo info = analyzeMatmul(matmul);
      matrices.push_back(info);
    }

    // 步骤3：计算总tile数
    int totalTiles = 0;
    for (const auto& mat : matrices) {
      totalTiles += mat.numTiles();
    }

    // 步骤4：决定调度策略
    ScheduleStrategy strategy = decideStrategy(matrices, totalTiles);

    // 步骤5：生成调度计划
    TileSchedule schedule = generateSchedule(matrices, strategy);

    // 步骤6：应用变换
    applySchedule(funcOp, schedule);
  }
};
```

### 3.2 矩阵分析

```cpp
struct MatrixInfo {
  linalg::MatmulOp op;
  int64_t M, N, K;          // 矩阵维度
  int64_t tilesM, tilesN;   // tile数量
  int64_t remainderM, remainderN;  // 余数

  int64_t numTiles() const {
    return tilesM * tilesN;
  }

  float paddingRatio() const {
    int64_t total = tilesM * tilesN * tileK_ * tileN_;
    int64_t actual = M * N;
    return 1.0f - (float)actual / total;
  }
};

MatrixInfo analyzeMatmul(linalg::MatmulOp matmul) {
  MatrixInfo info;
  info.op = matmul;

  // 获取维度
  auto lhsType = matmul.getInputs()[0].getType().cast<RankedTensorType>();
  auto rhsType = matmul.getInputs()[1].getType().cast<RankedTensorType>();

  info.M = lhsType.getShape()[0];
  info.K = lhsType.getShape()[1];
  info.N = rhsType.getShape()[1];

  // 计算tile数量
  info.tilesM = (info.M + tileK_ - 1) / tileK_;
  info.tilesN = (info.N + tileN_ - 1) / tileN_;
  info.remainderM = info.M % tileK_;
  info.remainderN = info.N % tileN_;

  return info;
}
```

### 3.3 调度策略选择

```cpp
enum class ScheduleStrategy {
  Simple,      // 单矩阵，直接选择前16个tile
  Batching,    // 多矩阵，轮询组合
  Optimized    // 智能调度，最小化padding
};

ScheduleStrategy decideStrategy(const SmallVector<MatrixInfo>& matrices,
                                int totalTiles) {
  // 单矩阵场景
  if (matrices.size() == 1) {
    return ScheduleStrategy::Simple;
  }

  // 多矩阵场景
  if (totalTiles < numTiles_) {
    // tile总数不足，简单batching
    return ScheduleStrategy::Batching;
  } else if (totalTiles > numTiles_) {
    // tile总数过多，需要智能选择
    return ScheduleStrategy::Optimized;
  } else {
    // 正好够，直接使用
    return ScheduleStrategy::Simple;
  }
}
```

### 3.4 调度计划生成

```cpp
struct TileDescriptor {
  linalg::MatmulOp sourceOp;
  int64_t offsetM, offsetN;
  int64_t sizeM, sizeN;
  bool isPadding;
};

struct TileSchedule {
  SmallVector<TileDescriptor, 16> tiles;
  int numRealTiles;
  int numPaddingTiles;
  float paddingRatio;
};

TileSchedule generateSchedule(const SmallVector<MatrixInfo>& matrices,
                              ScheduleStrategy strategy) {
  switch (strategy) {
    case ScheduleStrategy::Simple:
      return simpleSchedule(matrices[0]);
    case ScheduleStrategy::Batching:
      return batchingSchedule(matrices);
    case ScheduleStrategy::Optimized:
      return optimizedSchedule(matrices);
  }
}

TileSchedule optimizedSchedule(const SmallVector<MatrixInfo>& matrices) {
  TileSchedule schedule;

  // 收集所有候选tile
  struct TileCandidate {
    MatrixInfo* matrix;
    int64_t i, j;  // tile索引
    int64_t sizeM, sizeN;
    float paddingRatio;
  };

  SmallVector<TileCandidate> candidates;
  for (auto& mat : matrices) {
    for (int j = 0; j < mat.tilesN; j++) {
      for (int i = 0; i < mat.tilesM; i++) {
        TileCandidate cand;
        cand.matrix = &mat;
        cand.i = i;
        cand.j = j;
        cand.sizeM = std::min(tileK_, mat.M - i * tileK_);
        cand.sizeN = std::min(tileN_, mat.N - j * tileN_);
        cand.paddingRatio = 1.0f - (float)(cand.sizeM * cand.sizeN) /
                                    (tileK_ * tileN_);
        candidates.push_back(cand);
      }
    }
  }

  // 按padding排序
  std::sort(candidates.begin(), candidates.end(),
            [](const TileCandidate& a, const TileCandidate& b) {
              return a.paddingRatio < b.paddingRatio;
            });

  // 选择前16个
  for (int i = 0; i < numTiles_ && i < candidates.size(); i++) {
    const auto& cand = candidates[i];
    TileDescriptor desc;
    desc.sourceOp = cand.matrix->op;
    desc.offsetM = cand.i * tileK_;
    desc.offsetN = cand.j * tileN_;
    desc.sizeM = cand.sizeM;
    desc.sizeN = cand.sizeN;
    desc.isPadding = false;
    schedule.tiles.push_back(desc);
  }

  // 填充不足
  while (schedule.tiles.size() < numTiles_) {
    TileDescriptor padding;
    padding.isPadding = true;
    schedule.tiles.push_back(padding);
  }

  return schedule;
}
```

---

## 4. IR变换实现

### 4.1 应用调度（生成MLIR代码）

```cpp
void applySchedule(func::FuncOp funcOp, const TileSchedule& schedule) {
  OpBuilder builder(funcOp.getContext());

  // 在函数开始处插入
  Block& entryBlock = funcOp.getBody().front();
  builder.setInsertionPointToStart(&entryBlock);

  // 创建tile batch
  auto batchOp = builder.create<npu::TileBatchOp>(
    funcOp.getLoc(),
    builder.getI64IntegerAttr(numTiles_)
  );

  // 提取并添加所有tile
  for (int i = 0; i < schedule.tiles.size(); i++) {
    const auto& desc = schedule.tiles[i];

    if (desc.isPadding) {
      // 创建padding tile（全零）
      auto zeroTile = createPaddingTile(builder, funcOp.getLoc());
      builder.create<npu::AddToBatchOp>(
        funcOp.getLoc(),
        batchOp.getResult(),
        zeroTile,
        builder.getI64IntegerAttr(i),
        builder.getBoolAttr(true)
      );
    } else {
      // 从源张量提取tile
      Value input = desc.sourceOp.getInputs()[0];  // 假设是LHS

      SmallVector<Value> offsets;
      offsets.push_back(builder.create<arith::ConstantIndexOp>(
        funcOp.getLoc(), desc.offsetM));
      offsets.push_back(builder.create<arith::ConstantIndexOp>(
        funcOp.getLoc(), desc.offsetN));

      auto extractOp = builder.create<npu::ExtractTileOp>(
        funcOp.getLoc(),
        input,
        offsets,
        builder.getI64ArrayAttr({desc.sizeM, desc.sizeN})
      );

      builder.create<npu::AddToBatchOp>(
        funcOp.getLoc(),
        batchOp.getResult(),
        extractOp.getResult(),
        builder.getI64IntegerAttr(i),
        builder.getBoolAttr(false)
      );
    }
  }

  // 执行batch
  auto executeOp = builder.create<npu::ExecuteBatchOp>(
    funcOp.getLoc(),
    batchOp.getResult(),
    builder.getStringAttr("matmul")
  );

  // 提取结果
  for (int i = 0; i < schedule.tiles.size(); i++) {
    if (!schedule.tiles[i].isPadding) {
      auto resultOp = builder.create<npu::ExtractResultOp>(
        funcOp.getLoc(),
        executeOp.getResult(),
        builder.getI64IntegerAttr(i)
      );

      // 将结果写回原始张量
      // ... (使用tensor.insert_slice)
    }
  }
}
```

### 4.2 变换示例

**输入MLIR**：
```mlir
func.func @matmul_batch(
  %A0: tensor<33x48xf32>,
  %A1: tensor<50x30xf32>
) -> (tensor<33x48xf32>, tensor<50x30xf32>) {
  %C0 = linalg.matmul ins(%A0, %B0 : ...) outs(%Out0 : ...)
  %C1 = linalg.matmul ins(%A1, %B1 : ...) outs(%Out1 : ...)
  return %C0, %C1
}
```

**输出MLIR**（应用Pass后）：
```mlir
func.func @matmul_batch(...) -> (...) {
  // 创建16-tile batch
  %batch = npu.create_batch num_tiles = 16 : !npu.tile_batch<16>

  // 从A0提取tile
  %c0 = arith.constant 0 : index
  %tile0 = npu.extract_tile %A0[%c0, %c0] [16, 16]
    : tensor<33x48xf32> -> tensor<16x16xf32>
  npu.add_to_batch %batch, %tile0, slot 0, is_padding false

  %tile1 = npu.extract_tile %A0[%c16, %c0] [16, 16]
    : tensor<33x48xf32> -> tensor<16x16xf32>
  npu.add_to_batch %batch, %tile1, slot 1, is_padding false

  // 从A1提取tile
  %tile2 = npu.extract_tile %A1[%c0, %c0] [16, 16]
    : tensor<50x30xf32> -> tensor<16x16xf32>
  npu.add_to_batch %batch, %tile2, slot 2, is_padding false

  // ... 更多tile ...

  // 执行NPU计算
  %results = npu.execute_batch %batch, operation = "matmul"
    : !npu.tile_batch<16> -> !npu.result_batch<16>

  // 提取结果并组装
  %result0_tile0 = npu.extract_result %results, slot 0
    : !npu.result_batch<16> -> tensor<16x16xf32>
  %C0_partial = tensor.insert_slice %result0_tile0 into %C0[0, 0][16, 16][1, 1]
    : tensor<16x16xf32> into tensor<33x48xf32>

  // ... 更多结果组装 ...

  return %C0, %C1
}
```

---

## 5. 下一步Lowering

### 5.1 NPU Dialect → LLVM IR

```cpp
// Lower npu.execute_batch to LLVM intrinsic
class ExecuteBatchOpLowering : public ConvertOpToLLVMPattern<npu::ExecuteBatchOp> {
  LogicalResult matchAndRewrite(...) const override {
    // 生成NPU intrinsic调用
    auto intrinsic = rewriter.create<LLVM::InlineAsmOp>(
      op.getLoc(),
      /*asm=*/"npu_matmul_batch $0, $1, $2",
      /*constraints=*/"=r,r,r",
      /*operands=*/{batch_ptr, input_ptr, output_ptr}
    );

    return success();
  }
};
```

### 5.2 或者生成C++ Runtime调用

```cpp
// 生成对运行时库的调用
extern "C" {
  void npu_execute_matmul_batch(
    void* batch,
    int num_tiles,
    TileDescriptor* descriptors
  );
}
```

---

## 6. 测试用例

### 6.1 单元测试

```mlir
// RUN: mlir-opt %s -npu-tile-scheduler="num-tiles=16 tile-k=16 tile-n=16" | FileCheck %s

func.func @test_single_matrix(%A: tensor<97x137xf32>) -> tensor<97x137xf32> {
  %B = ...
  %C = linalg.matmul ins(%A, %B : ...) outs(...)
  return %C
}

// CHECK-LABEL: func @test_single_matrix
// CHECK: %[[BATCH:.*]] = npu.create_batch num_tiles = 16
// CHECK: npu.extract_tile
// CHECK-COUNT-16: npu.add_to_batch
// CHECK: npu.execute_batch %[[BATCH]]
```

### 6.2 集成测试

```cpp
// 测试不同策略
TEST(NPUTileScheduler, SimpleStrategy) {
  auto input = createMatrix(97, 137);
  auto schedule = scheduler.schedule({input});
  EXPECT_EQ(schedule.numRealTiles, 16);
  EXPECT_LT(schedule.paddingRatio, 0.15);
}

TEST(NPUTileScheduler, OptimizedStrategy) {
  auto matrices = {
    createMatrix(33, 48),
    createMatrix(50, 30),
    createMatrix(20, 25)
  };
  auto schedule = scheduler.schedule(matrices);
  EXPECT_EQ(schedule.numRealTiles, 16);
  EXPECT_LT(schedule.paddingRatio, 0.02);  // <2%
}
```

---

## 7. 实现优先级

### Phase 1: 基础Pass框架 ✓
- [ ] TableGen定义
- [ ] Pass注册
- [ ] 基本IR遍历

### Phase 2: 简单策略 ✓
- [ ] 单矩阵调度
- [ ] 边界处理
- [ ] Padding插入

### Phase 3: 优化策略 ✓
- [ ] 多矩阵batching
- [ ] 智能tile选择
- [ ] Padding最小化

### Phase 4: Lowering
- [ ] NPU Dialect → LLVM
- [ ] Runtime库接口
- [ ] 性能优化

---

## 8. 讨论要点

### 8.1 设计选择

**问题1**：是否需要自定义NPU Dialect？
- 选项A：使用现有Dialect（Linalg + SCF）
- 选项B：定义NPU Dialect（更清晰，推荐）✓

**问题2**：调度时机
- 选项A：编译时完全确定（推荐）✓
- 选项B：运行时动态调度（开销大）

**问题3**：Padding策略
- 选项A：填充NOP tile（硬件skip）✓
- 选项B：填充零矩阵（浪费计算）

### 8.2 扩展性

**支持更多操作**：
- Convolution (3×3 tile)
- Element-wise ops (1×1 tile)
- Reduction (特殊处理)

**支持动态shape**：
- 运行时padding
- 动态dispatch

---

## 9. 下一步行动

1. **实现基础Pass框架**（TableGen + C++骨架）
2. **定义NPU Dialect**（最小可用集）
3. **实现简单调度策略**（单矩阵场景）
4. **添加测试用例**（lit tests）
5. **集成到编译流程**（Pipeline配置）

---

**你的反馈**：
- 这个设计方案是否符合你的需求？
- 需要调整哪些部分？
- 优先实现哪个phase？

我们可以现在就开始实现Phase 1！🚀
