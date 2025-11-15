// RUN: mlir-opt %s -npu-tile-scheduler | FileCheck %s

// Test 1: Single matrix with irregular dimensions (K-loop unrolled)
// Matrix A[37×50] × B[50×64] → C[37×64]
// Expected: 4 K-iterations (0..16, 16..32, 32..48, 48..50)

module {
  func.func @matmul_37x50(%arg0: tensor<37x50xf32>, %arg1: tensor<50x64xf32>) -> tensor<37x64xf32> {
    // CHECK-LABEL: func.func @matmul_37x50

    // CHECK: %[[C0_F32:.*]] = arith.constant 0.000000e+00 : f32
    // CHECK: %[[C_INIT:.*]] = tensor.empty() : tensor<37x64xf32>
    // CHECK: %[[C_ZERO:.*]] = linalg.fill ins(%[[C0_F32]] : f32) outs(%[[C_INIT]] : tensor<37x64xf32>)

    // CHECK: // K-iteration 0: k=0..16
    // CHECK: %[[K0_OFFSET:.*]] = arith.constant 0 : index
    // CHECK: %[[K0_SIZE:.*]] = arith.constant 16 : index

    // CHECK: // Extract tiles for K-slice 0
    // CHECK: %[[TILE_A_0:.*]] = tensor.extract_slice %arg0
    // CHECK: %[[TILE_B_0:.*]] = tensor.extract_slice %arg1

    // CHECK: // Create batch for K-iteration 0
    // CHECK: %[[BATCH_0:.*]] = npu.create_batch : !npu.batch
    // CHECK: %[[BATCH_0_1:.*]] = npu.add_to_batch %[[BATCH_0]]

    // CHECK: // Execute batch 0
    // CHECK: %[[BATCH_RESULT_0:.*]] = npu.execute_batch %{{.*}} {operation = "matmul"}

    // CHECK: // Extract and accumulate results from K-iteration 0
    // CHECK: %[[TILE_RESULT_0:.*]] = npu.extract_result %[[BATCH_RESULT_0]]
    // CHECK: linalg.add
    // CHECK: tensor.insert_slice

    // CHECK: // K-iteration 1: k=16..32
    // CHECK: %[[K1_OFFSET:.*]] = arith.constant 16 : index
    // CHECK: %[[K1_SIZE:.*]] = arith.constant 16 : index

    // CHECK: // K-iteration 2: k=32..48
    // CHECK: %[[K2_OFFSET:.*]] = arith.constant 32 : index
    // CHECK: %[[K2_SIZE:.*]] = arith.constant 16 : index

    // CHECK: // K-iteration 3: k=48..50 (remainder)
    // CHECK: %[[K3_OFFSET:.*]] = arith.constant 48 : index
    // CHECK: %[[K3_SIZE:.*]] = arith.constant 2 : index

    // Original tosa.matmul should be removed
    %0 = tosa.matmul %arg0, %arg1 : (tensor<37x50xf32>, tensor<50x64xf32>) -> tensor<37x64xf32>

    // CHECK: return %{{.*}} : tensor<37x64xf32>
    return %0 : tensor<37x64xf32>
  }
}

// Test 2: Multiple matrices (multi-matrix optimization, both unrolled)
module {
  func.func @multi_matmul(
      %arg0: tensor<37x50xf32>, %arg1: tensor<50x64xf32>,
      %arg2: tensor<20x30xf32>, %arg3: tensor<30x40xf32>)
      -> (tensor<37x64xf32>, tensor<20x40xf32>) {
    // CHECK-LABEL: func.func @multi_matmul

    // Both matmuls should be transformed independently (unrolled)
    // Matrix 1: 4 K-iterations
    // Matrix 2: 2 K-iterations (K=30: 0..16, 16..30)

    // CHECK: linalg.fill
    // CHECK: // K-iteration 0
    // CHECK: npu.create_batch
    // CHECK: npu.execute_batch

    %0 = tosa.matmul %arg0, %arg1 : (tensor<37x50xf32>, tensor<50x64xf32>) -> tensor<37x64xf32>
    %1 = tosa.matmul %arg2, %arg3 : (tensor<20x30xf32>, tensor<30x40xf32>) -> tensor<20x40xf32>

    return %0, %1 : tensor<37x64xf32>, tensor<20x40xf32>
  }
}

// Test 3: Perfect tile alignment (no remainder, unrolled)
module {
  func.func @matmul_32x32(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-LABEL: func.func @matmul_32x32

    // K-dimension: 32 = 2 iterations of 16
    // M×N: 32×32 = 4 tiles (2×2)
    // Each batch: 4 tiles (all complete, no padding!)

    // K-iteration 0: k=0..16
    // CHECK: arith.constant 0 : index
    // CHECK: arith.constant 16 : index
    // CHECK: npu.create_batch
    // CHECK-COUNT-4: npu.add_to_batch
    // CHECK-NOT: npu.nop_tile
    // CHECK: npu.execute_batch

    // K-iteration 1: k=16..32
    // CHECK: arith.constant 16 : index
    // CHECK: npu.create_batch

    %0 = tosa.matmul %arg0, %arg1 : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    return %0 : tensor<32x32xf32>
  }
}

// Test 4: Small matrix (requires NOP filling, unrolled)
module {
  func.func @matmul_10x10(%arg0: tensor<10x10xf32>, %arg1: tensor<10x10xf32>) -> tensor<10x10xf32> {
    // CHECK-LABEL: func.func @matmul_10x10

    // K=10: only 1 K-iteration (k=0..10)
    // Only 1 real tile, need 15 NOPs
    // CHECK: arith.constant 0 : index
    // CHECK: arith.constant 10 : index
    // CHECK: npu.create_batch
    // CHECK: npu.add_to_batch
    // CHECK-COUNT-15: npu.nop_tile
    // CHECK-COUNT-15: npu.add_to_batch

    %0 = tosa.matmul %arg0, %arg1 : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
    return %0 : tensor<10x10xf32>
  }
}

// Test 5: Extreme irregular dimensions (9 K-iterations unrolled)
module {
  func.func @matmul_97x137(%arg0: tensor<97x137xf32>, %arg1: tensor<137x200xf32>) -> tensor<97x200xf32> {
    // CHECK-LABEL: func.func @matmul_97x137

    // K-dimension: 137 = 8×16 + 9, so 9 K iterations
    // M×N: 97×200 → 7×13 = 91 tiles per K-slice
    // Expected: 9 unrolled K-iterations

    // CHECK: linalg.fill

    // K-iteration 0: k=0..16
    // CHECK: arith.constant 0 : index
    // CHECK: arith.constant 16 : index
    // CHECK: npu.create_batch
    // CHECK: npu.execute_batch

    // K-iteration 8: k=128..137 (actualK=9, remainder!)
    // CHECK: arith.constant 128 : index
    // CHECK: arith.constant 9 : index

    %0 = tosa.matmul %arg0, %arg1 : (tensor<97x137xf32>, tensor<137x200xf32>) -> tensor<97x200xf32>
    return %0 : tensor<97x200xf32>
  }
}
