//===- NPUPasses.td - NPU Transformation Passes ----------*- tablegen -*-===//

#ifndef NPU_PASSES
#define NPU_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// NPU Tile Scheduler Pass
//===----------------------------------------------------------------------===//

def NPUTileSchedulerPass : Pass<"npu-tile-scheduler", "func::FuncOp"> {
  let summary = "Schedule TOSA matmul into NPU tile batches with K-reduction";
  let description = [{
    This pass transforms tosa.matmul operations into NPU dialect operations,
    scheduling tiles to maximize hardware utilization while handling K-dimension
    reduction correctly.

    Algorithm:
    1. Identify all tosa.matmul operations
    2. For each K-slice (K-dimension iteration):
       a. Generate M×N tile candidates
       b. Sort by padding ratio (minimize wasted computation)
       c. Select best 16 tiles per batch
       d. Generate NPU dialect IR
    3. Accumulate results across K iterations

    Features:
    - Supports arbitrary irregular matrix dimensions
    - Variable-length tiles (hardware-native support)
    - Global optimization across multiple matrices
    - K-dimension reduction with accumulation

    Example transformation (K-loop fully unrolled):
    ```mlir
    // Before
    %C = tosa.matmul %A, %B : (tensor<37x50xf32>, tensor<50x64xf32>) -> tensor<37x64xf32>

    // After (fully unrolled, no scf.for)
    // K-iteration 0: k=0..16
    %A_slice_0 = tensor.extract_slice %A[0, 0] [37, 16]
    %B_slice_0 = tensor.extract_slice %B[0, 0] [16, 64]
    %batch_0 = npu.create_batch
    // Add M×N tiles (12 real + 4 NOPs)
    %result_0 = npu.execute_batch %batch_0

    // K-iteration 1: k=16..32
    %A_slice_1 = tensor.extract_slice %A[0, 16] [37, 16]
    %B_slice_1 = tensor.extract_slice %B[16, 0] [16, 64]
    %batch_1 = npu.create_batch
    %result_1 = npu.execute_batch %batch_1

    // K-iteration 2, 3... (omitted for brevity)

    // Accumulate all K-slices
    %acc_1 = linalg.add %result_0, %result_1
    %acc_2 = linalg.add %acc_1, %result_2
    %C = linalg.add %acc_2, %result_3
    ```
  }];

  let constructor = "mlir::npu::createNPUTileSchedulerPass()";

  let options = [
    Option<"tileSize", "tile-size", "unsigned", /*default=*/"16",
           "Base tile size (default 16 for 16×16 tiles)">,
    Option<"maxTilesPerBatch", "max-tiles", "unsigned", /*default=*/"16",
           "Maximum tiles per batch (hardware constraint)">,
    Option<"enableMultiMatrixOptimization", "enable-multi-matrix", "bool",
           /*default=*/"true",
           "Enable global optimization across multiple matrices">
  ];

  let dependentDialects = [
    "mlir::tosa::TosaDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    // SCF removed: K-loop is fully unrolled, no scf.for needed
    "mlir::linalg::LinalgDialect",
    "mlir::npu::NPUDialect"
  ];
}

#endif // NPU_PASSES
