//===- BatchCompPasses.td - Batch Composition Passes -----*- tablegen -*-===//
//
// Pass definitions for Batch Composition Optimization
//
//===----------------------------------------------------------------------===//

#ifndef BATCHCOMP_PASSES
#define BATCHCOMP_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// BatchCompTileSchedulerPass
//===----------------------------------------------------------------------===//

def BatchCompTileSchedulerPass : Pass<"batch-comp-tile-scheduler", "func::FuncOp"> {
  let summary = "Optimize matmul operations with batch composition";

  let description = [{
    This pass transforms TOSA/Linalg matmul operations into BatchComp dialect
    operations with optimized tile-to-batch scheduling.

    The pass performs the following steps:
    1. Collect all matmul operations in the function
    2. Generate tiles for each matmul (considering tile size)
    3. Call Python optimizer to find optimal batch assignment
    4. Generate high-level BatchComp IR (generate_tiles, schedule_batches, execute_schedule)

    Example transformation:
    ```mlir
    // Input
    %C = tosa.matmul %A, %B : (tensor<100x200xf32>, tensor<200x300xf32>)

    // Output
    %tileset = batch_comp.generate_tiles %A, %B {tile_size = [16,16,16]}
    %schedule = batch_comp.schedule_batches %tileset {hardware = "npu"}
    %C = batch_comp.execute_schedule %schedule
    ```

    Options:
    - hardware: Target hardware platform ("npu", "gpu", "tpu")
    - tile-size: Tile dimensions [M, N, K] (default: [16, 16, 16])
    - algorithm: Optimization algorithm ("auto", "greedy", "improved_greedy", "cpsat")
    - config-path: Path to hardware configuration YAML file
  }];

  let constructor = "mlir::batch_comp::createBatchCompTileSchedulerPass()";

  let options = [
    Option<"hardware", "hardware", "std::string",
           /*default=*/"\"npu\"",
           "Target hardware platform (npu, gpu, tpu)">,

    ListOption<"tileSize", "tile-size", "int64_t",
               "Tile dimensions [M, N, K]">,

    Option<"algorithm", "algorithm", "std::string",
           /*default=*/"\"auto\"",
           "Optimization algorithm (auto, greedy, improved_greedy, cpsat)">,

    Option<"configPath", "config-path", "std::string",
           /*default=*/"\"\"",
           "Path to hardware configuration YAML file">,

    Option<"enableMultiMatmul", "enable-multi-matmul", "bool",
           /*default=*/"true",
           "Enable multi-matmul batch composition optimization">,
  ];

  let dependentDialects = [
    "mlir::tosa::TosaDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::batch_comp::BatchCompDialect"
  ];
}

//===----------------------------------------------------------------------===//
// BatchCompLoweringPass
//===----------------------------------------------------------------------===//

def BatchCompLoweringPass : Pass<"batch-comp-lowering", "func::FuncOp"> {
  let summary = "Lower high-level BatchComp IR to low-level BatchComp IR";

  let description = [{
    This pass lowers high-level BatchComp operations to low-level batch
    operations that can be directly translated to hardware instructions.

    Lowering rules:
    - generate_tiles → slice + pad operations
    - schedule_batches → create_batch + add_to_batch operations
    - execute_schedule → execute_batch + accumulate operations

    Example transformation:
    ```mlir
    // Input (high-level)
    %tileset = batch_comp.generate_tiles %A, %B {tile_size = [16,16,16]}
    %schedule = batch_comp.schedule_batches %tileset {hardware = "npu"}
    %C = batch_comp.execute_schedule %schedule

    // Output (low-level)
    %tile_a = batch_comp.slice %A[0, 0] [16, 16]
    %tile_b = batch_comp.slice %B[0, 0] [16, 16]
    %batch0 = batch_comp.create_batch {batch_size = 16}
    %batch1 = batch_comp.add_to_batch %batch0, %tile_a, %tile_b {...}
    %result = batch_comp.execute_batch %batch1 {operation = "matmul"}
    // ... accumulate results ...
    ```
  }];

  let constructor = "mlir::batch_comp::createBatchCompLoweringPass()";

  let options = [
    Option<"preserveHighLevel", "preserve-high-level", "bool",
           /*default=*/"false",
           "Preserve high-level ops for debugging">,
  ];

  let dependentDialects = [
    "mlir::tensor::TensorDialect",
    "mlir::arith::ArithDialect",
    "mlir::batch_comp::BatchCompDialect"
  ];
}

//===----------------------------------------------------------------------===//
// BatchCompToNPUPass
//===----------------------------------------------------------------------===//

def BatchCompToNPUPass : Pass<"batch-comp-to-npu", "func::FuncOp"> {
  let summary = "Lower BatchComp dialect to NPU dialect";

  let description = [{
    This pass translates BatchComp low-level operations to NPU-specific
    dialect operations.

    Mapping:
    - batch_comp.create_batch → npu.create_batch
    - batch_comp.add_to_batch → npu.add_to_batch
    - batch_comp.execute_batch → npu.execute_batch

    This is the final step before code generation.
  }];

  let constructor = "mlir::batch_comp::createBatchCompToNPUPass()";

  let dependentDialects = [
    "mlir::batch_comp::BatchCompDialect",
    "mlir::npu::NPUDialect"
  ];
}

//===----------------------------------------------------------------------===//
// Utility Passes
//===----------------------------------------------------------------------===//

def BatchCompCanonicalizePass : Pass<"batch-comp-canonicalize", "func::FuncOp"> {
  let summary = "Canonicalization patterns for BatchComp dialect";

  let description = [{
    Apply canonicalization patterns to simplify BatchComp IR:
    - Remove redundant slice operations
    - Merge adjacent pad operations
    - Eliminate NOPs from batches
    - Fold constant operations
  }];

  let constructor = "mlir::batch_comp::createBatchCompCanonicalizePass()";

  let dependentDialects = ["mlir::batch_comp::BatchCompDialect"];
}

#endif // BATCHCOMP_PASSES
