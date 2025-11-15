// ===== 完全展开的 K-loop 示例 =====
// 展示 NPU Tile Scheduler Pass 如何将 tosa.matmul 转换为完全展开的代码
// 特点：
// - 无 scf.for 循环
// - 每个 K-iteration 生成独立的代码块
// - 显式的 tensor.extract_slice 操作
// - 显式的 linalg.add 累加操作

// ========== 示例 1: 小矩阵 K=50 展开为 4 个 K-iterations ==========

// 输入: Matrix A[37×50] × B[50×64] → C[37×64]
module @unrolled_k50 {
  func.func @matmul_37x50(%A: tensor<37x50xf32>, %B: tensor<50x64xf32>) -> tensor<37x64xf32> {
    %C = tosa.matmul %A, %B : (tensor<37x50xf32>, tensor<50x64xf32>) -> tensor<37x64xf32>
    return %C : tensor<37x64xf32>
  }
}

// 预期输出 (展开后):
//
// func.func @matmul_37x50(%A: tensor<37x50xf32>, %B: tensor<50x64xf32>) -> tensor<37x64xf32> {
//   // Initialize accumulator with zeros
//   %c0_f32 = arith.constant 0.000000e+00 : f32
//   %C_init = tensor.empty() : tensor<37x64xf32>
//   %C_zero = linalg.fill ins(%c0_f32 : f32) outs(%C_init : tensor<37x64xf32>) -> tensor<37x64xf32>
//
//   // ===== K-iteration 0: k=0..16 (actualK=16) =====
//   %c0_0 = arith.constant 0 : index
//   %c16_0 = arith.constant 16 : index
//
//   // Extract M×N tiles for K-slice 0
//   // Tile (0,0): M=0..16, N=0..16, K=0..16
//   %tile_A_0_0_0 = tensor.extract_slice %A[0, 0] [16, 16] [1, 1] : tensor<37x50xf32> to tensor<16x16xf32>
//   %tile_B_0_0_0 = tensor.extract_slice %B[0, 0] [16, 16] [1, 1] : tensor<50x64xf32> to tensor<16x16xf32>
//
//   // Tile (0,16): M=0..16, N=16..32, K=0..16
//   %tile_A_0_16_0 = tensor.extract_slice %A[0, 0] [16, 16] [1, 1] : tensor<37x50xf32> to tensor<16x16xf32>
//   %tile_B_0_16_0 = tensor.extract_slice %B[0, 16] [16, 16] [1, 1] : tensor<50x64xf32> to tensor<16x16xf32>
//
//   // ... (更多的 M×N tiles，共 12 个实际 tiles)
//
//   // Tile (16,0): M=16..32, N=0..16
//   %tile_A_16_0_0 = tensor.extract_slice %A[16, 0] [16, 16] [1, 1] : tensor<37x50xf32> to tensor<16x16xf32>
//   %tile_B_16_0_0 = tensor.extract_slice %B[0, 0] [16, 16] [1, 1] : tensor<50x64xf32> to tensor<16x16xf32>
//
//   // Edge tile: M=21..37 (actualM=5), N=48..64 (actualN=16)
//   %tile_A_21_48_0 = tensor.extract_slice %A[21, 0] [5, 16] [1, 1] : tensor<37x50xf32> to tensor<5x16xf32>
//   %tile_B_21_48_0 = tensor.extract_slice %B[0, 48] [16, 16] [1, 1] : tensor<50x64xf32> to tensor<16x16xf32>
//
//   // Create batch for K-iteration 0
//   %batch_0 = npu.create_batch : !npu.batch
//   %batch_0_1 = npu.add_to_batch %batch_0, %tile_A_0_0_0, %tile_B_0_0_0 {
//     slot = 0 : i32, matrix_id = 0 : i32,
//     m_offset = 0 : i64, n_offset = 0 : i64, k_offset = 0 : i64,
//     actual_m = 16 : i64, actual_n = 16 : i64, actual_k = 16 : i64
//   } : !npu.batch, tensor<16x16xf32>, tensor<16x16xf32>
//
//   // ... (add 15 more tiles, total 16 with NOPs)
//
//   // Execute batch 0
//   %batch_result_0 = npu.execute_batch %batch_0_final {operation = "matmul"} : !npu.batch -> !npu.result
//
//   // Extract results and accumulate into C_zero
//   %tile_result_0_0_0 = npu.extract_result %batch_result_0 {slot = 0 : i32} : !npu.result -> tensor<16x16xf32>
//   %old_value_0_0_0 = tensor.extract_slice %C_zero[0, 0] [16, 16] [1, 1] : tensor<37x64xf32> to tensor<16x16xf32>
//   %empty_0_0_0 = tensor.empty() : tensor<16x16xf32>
//   %new_value_0_0_0 = linalg.add ins(%old_value_0_0_0, %tile_result_0_0_0 : tensor<16x16xf32>, tensor<16x16xf32>)
//                                  outs(%empty_0_0_0 : tensor<16x16xf32>) -> tensor<16x16xf32>
//   %C_acc_0_0_0 = tensor.insert_slice %new_value_0_0_0 into %C_zero[0, 0] [16, 16] [1, 1] : tensor<16x16xf32> into tensor<37x64xf32>
//
//   // ... (accumulate all 12 real tile results)
//   // %C_acc_0 = final accumulator after K-iteration 0
//
//   // ===== K-iteration 1: k=16..32 (actualK=16) =====
//   %c16_1 = arith.constant 16 : index
//   %c16_size_1 = arith.constant 16 : index
//
//   // Extract M×N tiles for K-slice 1 (K-offset = 16)
//   %tile_A_0_0_1 = tensor.extract_slice %A[0, 16] [16, 16] [1, 1] : tensor<37x50xf32> to tensor<16x16xf32>
//   %tile_B_0_0_1 = tensor.extract_slice %B[16, 0] [16, 16] [1, 1] : tensor<50x64xf32> to tensor<16x16xf32>
//
//   // ... (more tiles)
//
//   // Create batch for K-iteration 1
//   %batch_1 = npu.create_batch : !npu.batch
//   // ... add tiles ...
//   %batch_result_1 = npu.execute_batch %batch_1_final {operation = "matmul"} : !npu.batch -> !npu.result
//
//   // Accumulate results into C_acc_0
//   // ... (similar accumulation as K-iteration 0)
//   // %C_acc_1 = final accumulator after K-iteration 1
//
//   // ===== K-iteration 2: k=32..48 (actualK=16) =====
//   %c32_2 = arith.constant 32 : index
//   %c16_size_2 = arith.constant 16 : index
//
//   // ... (similar to K-iteration 1)
//   // %C_acc_2 = final accumulator after K-iteration 2
//
//   // ===== K-iteration 3: k=48..50 (actualK=2, 注意余数!) =====
//   %c48_3 = arith.constant 48 : index
//   %c2_size_3 = arith.constant 2 : index  // Last K-slice only has 2 elements!
//
//   // Extract M×N tiles for K-slice 3 (K dimension = 2)
//   %tile_A_0_0_3 = tensor.extract_slice %A[0, 48] [16, 2] [1, 1] : tensor<37x50xf32> to tensor<16x2xf32>
//   %tile_B_0_0_3 = tensor.extract_slice %B[48, 0] [2, 16] [1, 1] : tensor<50x64xf32> to tensor<2x16xf32>
//
//   // ... (more tiles)
//
//   %batch_3 = npu.create_batch : !npu.batch
//   // ... add tiles with actual_k = 2 ...
//   %batch_result_3 = npu.execute_batch %batch_3_final {operation = "matmul"} : !npu.batch -> !npu.result
//
//   // Final accumulation
//   // ... accumulate into C_acc_2 ...
//   // %C_final = result after all 4 K-iterations
//
//   return %C_final : tensor<37x64xf32>
// }

// ========== 优点 ==========
// 1. 无循环控制开销 (no scf.for)
// 2. 完全静态展开，编译器可以完全优化
// 3. 清晰的数据流依赖
// 4. 便于后续优化:
//    - 并行执行多个 K-iteration (如果硬件支持)
//    - 融合 slice 和 batch 操作
//    - 死代码消除 (对于 NOP tiles)
//    - 常量折叠

// ========== 缺点 ==========
// 1. 代码体积大 (对于大 K 值)
//    - K=137 会生成 9 个 K-iterations
//    - 每个 iteration 可能有几十到上百行代码
// 2. 不适合动态 K 值 (需要 scf.for)

// ========== 示例 2: 完美对齐的矩阵 ==========

module @unrolled_aligned {
  func.func @matmul_32x32(%A: tensor<32x32xf32>, %B: tensor<32x32xf32>) -> tensor<32x32xf32> {
    %C = tosa.matmul %A, %B : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %C : tensor<32x32xf32>
  }
}

// 预期输出特点:
// - K=32 → 2 iterations (0..16, 16..32)
// - M×N = 2×2 = 4 tiles (all complete 16×16)
// - NO NOP tiles needed!
// - 100% hardware utilization

// K-iteration 0: 4 complete tiles
// K-iteration 1: 4 complete tiles
// Total: 8 tile computations + 8 accumulation ops

// ========== 运行命令 ==========
//
// 1. 运行 Pass 生成展开代码:
// $ mlir-opt unrolled_example.mlir -npu-tile-scheduler -o unrolled_output.mlir
//
// 2. 查看生成的 MLIR (会很长!):
// $ cat unrolled_output.mlir | less
//
// 3. 验证正确性:
// $ mlir-opt unrolled_output.mlir -verify-diagnostics
//
// 4. 代码大小分析:
// $ wc -l unrolled_output.mlir  # 查看生成的代码行数
