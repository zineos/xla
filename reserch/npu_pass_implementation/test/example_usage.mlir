// ===== 完整使用示例 =====
// 展示如何使用NPU Tile Scheduler Pass处理tosa.matmul

// ========== 示例1: 基本用法 - 不规则矩阵 ==========

// 输入MLIR（使用tosa.matmul）
module @example1 {
  func.func @matmul_irregular(%A: tensor<37x50xf32>, %B: tensor<50x64xf32>) -> tensor<37x64xf32> {
    %C = tosa.matmul %A, %B : (tensor<37x50xf32>, tensor<50x64xf32>) -> tensor<37x64xf32>
    return %C : tensor<37x64xf32>
  }
}

// 运行Pass:
// $ mlir-opt example_usage.mlir -npu-tile-scheduler

// 预期输出（简化版）：
//
// func.func @matmul_irregular(%A, %B) -> tensor<37x64xf32> {
//   %c0 = arith.constant 0
//   %C_zero = linalg.fill ...
//
//   // K-loop: 4 iterations (K=50: 0-16, 16-32, 32-48, 48-50)
//   %result = scf.for %k = 0 to 50 step 16 iter_args(%acc = %C_zero) {
//     %batch = npu.create_batch
//
//     // Batch contains 12 real tiles + 4 NOPs
//     // Real tiles (sorted by padding):
//     // - 6 complete 16×16 tiles (padding=0.0)
//     // - 3 edge 5×16 tiles (padding=0.688)
//     // - 2 edge 16×2 tiles (padding=0.875)
//     // - 1 corner 5×2 tile (padding=0.961)
//
//     %result = npu.execute_batch %batch
//     // Accumulate results
//     %acc_updated = ...
//     scf.yield %acc_updated
//   }
//   return %result
// }

// ========== 示例2: 完美对齐 ==========

module @example2 {
  func.func @matmul_aligned(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %C = tosa.matmul %A, %B : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %C : tensor<32x32xf32>
  }
}

// 预期输出特点：
// - K-loop: 2 iterations (K=32: 0-16, 16-32)
// - Each batch: 4 complete 16×16 tiles (M=2, N=2)
// - NO NOP tiles needed
// - 100% hardware utilization

// ========== 示例3: 多矩阵 ==========

module @example3 {
  func.func @multi_matmul(
      %A1: tensor<37x50xf32>, %B1: tensor<50x64xf32>,
      %A2: tensor<20x30xf32>, %B2: tensor<30x40xf32>)
      -> (tensor<37x64xf32>, tensor<20x40xf32>) {

    %C1 = tosa.matmul %A1, %B1 : (tensor<37x50xf32>, tensor<50x64xf32>) -> tensor<37x64xf32>
    %C2 = tosa.matmul %A2, %B2 : (tensor<20x30xf32>, tensor<30x40xf32>) -> tensor<20x40xf32>

    return %C1, %C2 : tensor<37x64xf32>, tensor<20x40xf32>
  }
}

// 预期输出特点：
// - 两个独立的K-loop（每个matmul一个）
// - Matrix 1: 4 K-iterations, 12 tiles per batch
// - Matrix 2: 2 K-iterations, 4 tiles per batch
//
// 注：当前实现独立处理，未来可以全局优化（合并同K-slice的tile）

// ========== 示例4: 极端不规则 ==========

module @example4 {
  func.func @matmul_extreme(%A: tensor<97x137xf32>, %B: tensor<137x200xf32>) -> tensor<97x200xf32> {
    %C = tosa.matmul %A, %B : (tensor<97x137xf32>, tensor<137x200xf32>) -> tensor<97x200xf32>
    return %C : tensor<97x200xf32>
  }
}

// 分析：
// - M = 97 → 7 tiles (6×16 + 1×1)
// - K = 137 → 9 iterations (8×16 + 1×9)
// - N = 200 → 13 tiles (12×16 + 1×8)
// - Total M×N tiles per K-slice: 7×13 = 91 tiles
// - Batches per K-iteration: 6 batches (16+16+16+16+16+11)
//
// 预期输出特点：
// - 9 K-iterations
// - 54 total batches (9 K × 6 batches)
// - Last batch of each K-iteration: 11 real tiles + 5 NOPs

// ========== 示例5: 小矩阵（需要大量NOP）==========

module @example5 {
  func.func @matmul_small(%A: tensor<10x10xf32>, %B: tensor<10x10xf32>) -> tensor<10x10xf32> {
    %C = tosa.matmul %A, %B : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    return %C : tensor<10x10xf32>
  }
}

// 分析：
// - Only 1 tile (10×10)
// - Need 15 NOP tiles
// - Utilization: 100/4096 = 2.4% (very low!)
//
// 优化建议：合并多个小矩阵到同一batch

// ========== 验证命令 ==========

// 1. 运行Pass
// $ mlir-opt example_usage.mlir -npu-tile-scheduler -o output.mlir

// 2. 查看生成的IR
// $ cat output.mlir

// 3. 验证IR正确性
// $ mlir-opt output.mlir -verify-diagnostics

// 4. 运行所有测试
// $ cd build && make check-npu
