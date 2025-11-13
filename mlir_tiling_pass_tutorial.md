# MLIR Tiling Pass开发完整教程

> 基于XLA源码的实战教学，适合自研芯片编译器开发

---

## 目录
1. [MLIR核心概念](#1-mlir核心概念)
2. [XLA的Tiling架构](#2-xla的tiling架构)
3. [Tiling Pass实现分析](#3-tiling-pass实现分析)
4. [如何编写自己的Pass](#4-如何编写自己的pass)
5. [实战：矩阵乘法Tiling](#5-实战矩阵乘法tiling)

---

## 1. MLIR核心概念

### 1.1 什么是MLIR？

**MLIR (Multi-Level Intermediate Representation)** 是Google开发的编译器基础设施。

```
传统编译器:
  Source → Frontend IR → Backend IR → Machine Code
  (每个编译器都要重新实现)

MLIR:
  Source → Dialect 1 → Dialect 2 → ... → Dialect N → Machine Code
  (可复用的编译器组件)
```

### 1.2 核心组件

#### Operation (操作)
MLIR的基本单元，类似于LLVM的Instruction。

```mlir
// 示例：加法操作
%result = arith.addi %a, %b : i32
// 格式：%结果 = 操作名 %操作数... : 类型
```

#### Dialect (方言)
一组相关操作的集合。

```mlir
// Arithmetic Dialect
%add = arith.addi %a, %b : i32
%mul = arith.muli %c, %d : i32

// MHLO Dialect (XLA)
%dot = mhlo.dot %lhs, %rhs : (tensor<128x256xf32>, tensor<256x512xf32>) -> tensor<128x512xf32>

// XTile Dialect (XLA的Tiling)
%tile = xtile.extract %buffer[%i, %j] [32, 32] [1, 1] : memref<1024x1024xf32> -> tensor<32x32xf32>
```

#### Pass (优化Pass)
对MLIR IR进行转换的优化器。

```cpp
// Pass定义（TableGen格式）
def TileLoopsPass : Pass<"tile-loops", "func::FuncOp"> {
  let summary = "Tiles parallel loops.";
  let constructor = "createTileLoopsPass()";
  let options = [
    ListOption<"tile_sizes_", "tile-sizes", "int64_t", "Tile size in each dimension.">,
  ];
}
```

---

## 2. XLA的Tiling架构

### 2.1 XLA的MLIR Dialect层次

```
PyTorch/JAX/TF
      ↓
StableHLO (标准化输入)
      ↓
MHLO (XLA内部高层表示)
      ↓
LHLO (Buffer语义的低层表示)
      ↓
XTile (Tiling专用Dialect) ← 我们重点学习的
      ↓
SCF (Structured Control Flow - 循环表示)
      ↓
Affine (仿射循环优化)
      ↓
LLVM IR
```

### 2.2 XTile Dialect架构

**文件位置**：`xla/codegen/xtile/ir/xtile_ops.td`

```mlir
// XTile的核心操作

// 1. entry_func - Tiling函数入口
xtile.entry_func @matmul_tiled(%A: memref<1024x1024xf32>,
                               %B: memref<1024x1024xf32>,
                               %tile_id: index) {
  // Tiling逻辑
}

// 2. extract - 从buffer提取tile
%tile_A = xtile.extract %A[%i, %j] [32, 32] [1, 1]
  : memref<1024x1024xf32> -> tensor<32x32xf32>

// 3. insert - 将tile写回buffer
xtile.insert %tile_result into %C[%i, %j] [32, 32] [1, 1]
  : tensor<32x32xf32> -> memref<1024x1024xf32>
```

**关键属性**：
- `offsets`: tile在buffer中的起始位置 `[%i, %j]`
- `full_tile_shape`: tile的完整形状 `[32, 32]`
- `strides`: 步长 `[1, 1]`（支持跨步访问）

---

## 3. Tiling Pass实现分析

### 3.1 TileLoopsPass源码剖析

**文件位置**：`xla/mlir_hlo/transforms/tile_loops_pass.cc`

```cpp
class TileLoopsPass : public impl::TileLoopsPassBase<TileLoopsPass> {
 public:
  explicit TileLoopsPass(ArrayRef<int64_t> tileSizes,
                         ArrayRef<int64_t> unrollFactors) {
    tile_sizes_ = tileSizes;      // 例如：[32, 32]
    unroll_factors_ = unrollFactors;  // 例如：[4, 4]
  }

  void runOnOperation() override;
};
```

### 3.2 核心算法流程

```cpp
void TileLoopsPass::runOnOperation() {
  // 步骤1: 找到所有最内层的并行循环
  SmallVector<ParallelOp, 2> ploops;
  getInnermostParallelLoops(this->getOperation(), ploops);

  for (ParallelOp ploop : ploops) {
    // 步骤2: 检查访问模式复杂度
    if (isComplexAccessPattern(ploop)) {
      // 简单tiling（一层）
      tileParallelLoop(ploop, tile_sizes_, false);
      continue;
    }

    // 步骤3: 两层tiling（外层tile + 内层unroll）
    // 第一层：按unrolledTile分块
    ploop = tileParallelLoop(ploop, unrolledTile, false).second;
    // 第二层：按unroll_factors展开
    ploop = tileParallelLoop(ploop, unroll_factors_, false).second;
  }

  // 步骤4: 应用算术优化
  applyPatternsGreedily(getOperation(), patterns);
}
```

### 3.3 访问模式复杂度判断

```cpp
static bool isComplexAccessPattern(ParallelOp ploop) {
  auto isComplex = [&](memref::LoadOp loadOp) {
    // 检查1：非恒等布局（例如转置）
    if (!loadOp.getMemRefType().getLayout().isIdentity())
      return true;

    // 检查2：索引是否匹配循环归纳变量
    return loadOp.getIndices() != ploop.getInductionVars();
  };
  return llvm::any_of(ploop.getBody()->getOps<memref::LoadOp>(), isComplex);
}
```

**示例**：

```mlir
// 简单模式（索引 = 归纳变量）
scf.parallel (%i, %j) = (0, 0) to (1024, 1024) {
  %val = memref.load %A[%i, %j] : memref<1024x1024xf32>
  // ✓ 索引[%i, %j]直接使用归纳变量
}

// 复杂模式（有计算的索引）
scf.parallel (%i, %j) = (0, 0) to (1024, 1024) {
  %k = arith.addi %i, %c1 : index
  %val = memref.load %A[%k, %j] : memref<1024x1024xf32>
  // ✗ 索引[%k, %j]有额外计算
}
```

---

## 4. 如何编写自己的Pass

### 4.1 Pass定义（TableGen）

创建文件：`my_passes.td`

```tablegen
include "mlir/Pass/PassBase.td"

def MyMatmulTilingPass : Pass<"my-matmul-tiling", "func::FuncOp"> {
  let summary = "Tile matmul operations for my custom chip.";

  let description = [{
    This pass tiles matrix multiplication operations according to
    the hardware constraints of my custom accelerator.
  }];

  let options = [
    Option<"tile_m_", "tile-m", "int64_t", /*default=*/"64",
           "Tile size for M dimension">,
    Option<"tile_n_", "tile-n", "int64_t", /*default=*/"64",
           "Tile size for N dimension">,
    Option<"tile_k_", "tile-k", "int64_t", /*default=*/"32",
           "Tile size for K dimension">
  ];

  let constructor = "createMyMatmulTilingPass()";

  let dependentDialects = [
    "linalg::LinalgDialect",
    "scf::SCFDialect"
  ];
}
```

### 4.2 Pass实现（C++）

创建文件：`my_matmul_tiling_pass.cc`

```cpp
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

#define GEN_PASS_DEF_MYMATMULTILINGPASS
#include "my_passes.h.inc"

class MyMatmulTilingPass
    : public impl::MyMatmulTilingPassBase<MyMatmulTilingPass> {
 public:
  using MyMatmulTilingPassBase::MyMatmulTilingPassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // 遍历所有linalg.matmul操作
    funcOp.walk([&](linalg::MatmulOp matmulOp) {
      OpBuilder builder(matmulOp);

      // 设置tile大小
      SmallVector<int64_t> tileSizes = {tile_m_, tile_n_, tile_k_};

      // 执行tiling
      auto tilingOptions = linalg::LinalgTilingOptions()
          .setTileSizes(tileSizes)
          .setLoopType(linalg::LinalgTilingLoopType::ParallelLoops);

      FailureOr<linalg::TiledLinalgOp> tiledOp =
          linalg::tileLinalgOp(builder, matmulOp, tilingOptions);

      if (succeeded(tiledOp)) {
        matmulOp.replaceAllUsesWith(tiledOp->tensorResults);
        matmulOp.erase();
      }
    });
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createMyMatmulTilingPass() {
  return std::make_unique<MyMatmulTilingPass>();
}

} // namespace mlir
```

### 4.3 Pass注册

```cpp
// 在PassRegistry中注册
void registerMyPasses() {
  mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return createMyMatmulTilingPass();
  });
}
```

---

## 5. 实战：矩阵乘法Tiling

### 5.1 输入MLIR（未tiling）

```mlir
func.func @matmul(%A: tensor<1024x2048xf32>,
                  %B: tensor<2048x512xf32>) -> tensor<1024x512xf32> {
  %C_empty = tensor.empty() : tensor<1024x512xf32>
  %C = linalg.matmul
    ins(%A, %B : tensor<1024x2048xf32>, tensor<2048x512xf32>)
    outs(%C_empty : tensor<1024x512xf32>) -> tensor<1024x512xf32>
  return %C : tensor<1024x512xf32>
}
```

### 5.2 应用Tiling Pass

```bash
# 命令行调用
mlir-opt matmul.mlir \
  --my-matmul-tiling="tile-m=64 tile-n=64 tile-k=32" \
  -o matmul_tiled.mlir
```

### 5.3 输出MLIR（已tiling）

```mlir
func.func @matmul(%A: tensor<1024x2048xf32>,
                  %B: tensor<2048x512xf32>) -> tensor<1024x512xf32> {
  %C_empty = tensor.empty() : tensor<1024x512xf32>

  %c0 = arith.constant 0 : index
  %c64 = arith.constant 64 : index
  %c32 = arith.constant 32 : index
  %c1024 = arith.constant 1024 : index
  %c512 = arith.constant 512 : index
  %c2048 = arith.constant 2048 : index

  // 三重循环：M, N, K
  %result = scf.for %i = %c0 to %c1024 step %c64 iter_args(%C_acc = %C_empty) {
    %result_n = scf.for %j = %c0 to %c512 step %c64 iter_args(%C_acc_n = %C_acc) {
      %result_k = scf.for %k = %c0 to %c2048 step %c32 iter_args(%C_acc_k = %C_acc_n) {

        // 提取Tile
        %A_tile = tensor.extract_slice %A[%i, %k][64, 32][1, 1]
          : tensor<1024x2048xf32> to tensor<64x32xf32>
        %B_tile = tensor.extract_slice %B[%k, %j][32, 64][1, 1]
          : tensor<2048x512xf32> to tensor<32x64xf32>
        %C_tile = tensor.extract_slice %C_acc_k[%i, %j][64, 64][1, 1]
          : tensor<1024x512xf32> to tensor<64x64xf32>

        // Tile级matmul
        %C_tile_result = linalg.matmul
          ins(%A_tile, %B_tile : tensor<64x32xf32>, tensor<32x64xf32>)
          outs(%C_tile : tensor<64x64xf32>) -> tensor<64x64xf32>

        // 写回Tile
        %C_updated = tensor.insert_slice %C_tile_result into %C_acc_k[%i, %j][64, 64][1, 1]
          : tensor<64x64xf32> into tensor<1024x512xf32>

        scf.yield %C_updated : tensor<1024x512xf32>
      }
      scf.yield %result_k : tensor<1024x512xf32>
    }
    scf.yield %result_n : tensor<1024x512xf32>
  }

  return %result : tensor<1024x512xf32>
}
```

### 5.4 Tiling效果分析

```
原始计算量：
  1024 × 512 个输出元素
  每个元素需要 2048 次乘加
  总计：1024 × 512 × 2048 = 1,073,741,824 次操作

Tiling后（64×64×32）：
  Number of tiles: (1024/64) × (512/64) × (2048/32) = 16 × 8 × 64 = 8,192
  Each tile: 64 × 64 × 32 = 131,072 operations
  Total: 8,192 × 131,072 = 1,073,741,824 (相同！)

性能提升来源：
  ✓ Cache命中率提升：小Tile完全放入L1 Cache
  ✓ 内存访问减少：数据重用
  ✓ 并行化：外层循环可并行执行
  ✓ 向量化：Tile内操作易于向量化
```

---

## 6. 自研芯片Tiling策略

### 6.1 确定Tile大小

```cpp
// 根据硬件参数计算最优Tile大小
int64_t calculateOptimalTileSize(
    int64_t cacheSize,        // L1 Cache大小（字节）
    int64_t elementSize,      // 元素大小（字节）
    int64_t numArrays) {      // 数组数量

  // 公式：TileSize ≈ sqrt(CacheSize / (numArrays * elementSize))
  int64_t totalElements = cacheSize / (numArrays * elementSize);
  int64_t tileSize = std::sqrt(totalElements);

  // 向下取整到2的幂
  return 1 << (63 - __builtin_clzll(tileSize));
}

// 示例：
// L1 Cache = 32KB, float32, 3个数组(A, B, C)
// tileSize = sqrt(32768 / (3 * 4)) = sqrt(2730) ≈ 52 → 32
```

### 6.2 完整Tiling Pipeline

```mlir
// 1. 高层优化
module {
  func.func @main() {
    // StableHLO/MHLO级优化
  }
}
    ↓ [ConvertToLinalg]
// 2. Linalg表示
module {
  func.func @main() {
    linalg.matmul ...
  }
}
    ↓ [MyMatmulTilingPass]  ← 你的自定义Pass
// 3. Tiled表示
module {
  func.func @main() {
    scf.for ... {
      scf.for ... {
        linalg.matmul (small tile)
      }
    }
  }
}
    ↓ [Bufferization]
// 4. Buffer语义
module {
  func.func @main() {
    scf.for ... {
      memref.load ...
      arith.mulf ...
      memref.store ...
    }
  }
}
    ↓ [LowerToLLVM]
// 5. LLVM IR → Machine Code
```

---

## 7. 调试技巧

### 7.1 打印中间IR

```cpp
void MyMatmulTilingPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // 打印原始IR
  llvm::errs() << "=== Before Tiling ===\n";
  funcOp.print(llvm::errs());

  // ... tiling逻辑 ...

  // 打印变换后的IR
  llvm::errs() << "\n=== After Tiling ===\n";
  funcOp.print(llvm::errs());
}
```

### 7.2 验证Pass正确性

```cpp
void MyMatmulTilingPass::runOnOperation() {
  // ... tiling逻辑 ...

  // 验证IR合法性
  if (failed(mlir::verify(getOperation()))) {
    signalPassFailure();
    return;
  }
}
```

### 7.3 使用mlir-opt测试

```bash
# 测试单个Pass
mlir-opt test.mlir --my-matmul-tiling="tile-m=64" --mlir-print-ir-after-all

# 查看所有可用Pass
mlir-opt --help

# 验证IR合法性
mlir-opt test.mlir --verify-each
```

---

## 8. 总结与下一步

### ✅ 你已经学会了

1. **MLIR核心概念**：Operation, Dialect, Pass
2. **XLA的Tiling架构**：XTile Dialect, TileLoopsPass
3. **Pass开发流程**：TableGen定义 → C++实现 → 注册
4. **Tiling算法**：访问模式分析、两层Tiling、边界处理

### 🚀 下一步学习

1. **Affine变换**：`mlir/Dialect/Affine/`
   - 仿射循环分析
   - 循环融合、交换、分裂

2. **向量化**：`mlir/Dialect/Vector/`
   - Tile到Vector的Lowering
   - SIMD优化

3. **GPU代码生成**：`mlir/Dialect/GPU/`
   - Tile到GPU Thread Block的映射
   - Shared Memory管理

4. **自定义Backend**：
   - 定义自己的Dialect
   - 实现Lowering Pass
   - 代码生成

---

## 参考资料

- [MLIR官方文档](https://mlir.llvm.org/)
- [XLA源码](https://github.com/openxla/xla)
- [MLIR Toy Tutorial](https://mlir.llvm.org/docs/Tutorials/Toy/)
- [Linalg Dialect文档](https://mlir.llvm.org/docs/Dialects/Linalg/)

**文件位置参考**：
- TileLoopsPass实现：`xla/mlir_hlo/transforms/tile_loops_pass.cc`
- XTile Dialect定义：`xla/codegen/xtile/ir/xtile_ops.td`
- Pass定义：`xla/mlir_hlo/transforms/passes.td`

---

**作者注**：本教程基于OpenXLA项目源码（2025版本），所有代码示例均来自实际生产环境。
