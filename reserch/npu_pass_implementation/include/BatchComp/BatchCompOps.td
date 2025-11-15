//===- BatchCompOps.td - Batch Composition Dialect Ops ---*- tablegen -*-===//
//
// Batch Composition Optimization Dialect for Tile-based Accelerators
//
// This dialect provides a unified abstraction for optimizing matrix operations
// across different hardware platforms (NPU, GPU, TPU) by treating the problem
// as batch composition optimization.
//
//===----------------------------------------------------------------------===//

#ifndef BATCHCOMP_OPS
#define BATCHCOMP_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"

//===----------------------------------------------------------------------===//
// Batch Composition Dialect
//===----------------------------------------------------------------------===//

def BatchComp_Dialect : Dialect {
  let name = "batch_comp";
  let summary = "Batch composition optimization for tile-based accelerators";
  let description = [{
    This dialect provides a unified abstraction for optimizing tile-based
    matrix computations across different hardware platforms.

    **Supported Hardware**:
    - NPU: Fixed-size batches (e.g., 16 tiles)
    - GPU: Warp-based execution (e.g., 32 threads)
    - TPU: Systolic arrays (e.g., 128×128)

    **Key Concepts**:
    - **Tile**: A small sub-matrix computation
    - **Batch**: A group of tiles executed together
    - **K-reduction**: Accumulation across K-dimension iterations

    **Problem Formulation**:
    The core optimization is a bin packing problem: assign tiles to batches
    to minimize batch count and maximize hardware utilization.

    Example transformation:
    ```mlir
    // Input: TOSA matmul
    %C = tosa.matmul %A, %B : (tensor<100x200xf32>, tensor<200x300xf32>)

    // Output: Batch composition
    %tiles = batch_comp.generate_tiles %A, %B {tile_size = [16,16,16]}
    %batches = batch_comp.schedule_batches %tiles {hardware = "npu"}
    %C = batch_comp.execute_batches %batches
    ```
  }];
  let cppNamespace = "::mlir::batch_comp";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Batch Composition Types
//===----------------------------------------------------------------------===//

class BatchComp_Type<string name, string typeMnemonic>
    : TypeDef<BatchComp_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def BatchComp_TileType : BatchComp_Type<"Tile", "tile"> {
  let summary = "A single tile computation";
  let description = [{
    Represents a small sub-matrix computation.
    Contains metadata about source location, size, and computation properties.
  }];
}

def BatchComp_TileSet : BatchComp_Type<"TileSet", "tileset"> {
  let summary = "A collection of tiles";
  let description = [{
    Represents multiple tiles to be scheduled into batches.
  }];
}

def BatchComp_Batch : BatchComp_Type<"Batch", "batch"> {
  let summary = "A batch of tiles";
  let description = [{
    Represents a group of tiles that execute together on hardware.
    The batch size is hardware-dependent (e.g., 16 for NPU, 32 for GPU).
  }];
}

def BatchComp_BatchSchedule : BatchComp_Type<"BatchSchedule", "schedule"> {
  let summary = "Complete batch schedule";
  let description = [{
    Represents the complete assignment of all tiles to batches.
  }];
}

//===----------------------------------------------------------------------===//
// Base Operation Class
//===----------------------------------------------------------------------===//

class BatchComp_Op<string mnemonic, list<Trait> traits = []> :
  Op<BatchComp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Tile Generation Operations
//===----------------------------------------------------------------------===//

def BatchComp_SliceOp : BatchComp_Op<"slice", [Pure]> {
  let summary = "Extract a tile slice from a tensor";
  let description = [{
    Extracts a sub-tensor slice for tile computation.

    This operation handles both regular and edge tiles (with padding if needed).

    Example:
    ```mlir
    // Extract 16×16 tile from A starting at [32, 64]
    %tile_a = batch_comp.slice %A[32, 64] [16, 16]
              : tensor<100x200xf32> -> tensor<16x16xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$source,
    Variadic<Index>:$offsets,
    Variadic<Index>:$sizes,
    DefaultValuedAttr<BoolAttr, "false">:$needs_padding
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $source `[` $offsets `]` `[` $sizes `]` attr-dict
    `:` type($source) `->` type($result)
  }];
}

def BatchComp_PadOp : BatchComp_Op<"pad", [Pure]> {
  let summary = "Pad a tile to target size";
  let description = [{
    Pads an edge tile to the target tile size.

    This is necessary when matrix dimensions are not divisible by tile size.
    Padding strategy depends on hardware:
    - NPU: Zero padding
    - GPU: Predicated execution (no actual padding)
    - TPU: Zero padding

    Example:
    ```mlir
    // Pad 13×16 edge tile to 16×16
    %padded = batch_comp.pad %tile to [16, 16] {value = 0.0 : f32}
              : tensor<13x16xf32> -> tensor<16x16xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$input,
    Variadic<Index>:$target_sizes,
    AnyAttr:$pad_value
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $input `to` `[` $target_sizes `]` attr-dict
    `:` type($input) `->` type($result)
  }];
}

def BatchComp_TileMatmulOp : BatchComp_Op<"tile_matmul", [Pure]> {
  let summary = "Compute tile-level matrix multiplication";
  let description = [{
    Performs a single tile matmul: C[m,n] = A[m,k] × B[k,n]

    This is the fundamental computation unit. Multiple tile_matmul operations
    with the same output location but different K-slices will be accumulated.

    Example:
    ```mlir
    %C_tile = batch_comp.tile_matmul %A_tile, %B_tile {
      matrix_id = 0 : i32,
      m_offset = 0 : i64,
      n_offset = 0 : i64,
      k_iteration = 0 : i64
    } : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    OptionalAttr<I32Attr>:$matrix_id,
    OptionalAttr<I64Attr>:$m_offset,
    OptionalAttr<I64Attr>:$n_offset,
    OptionalAttr<I64Attr>:$k_iteration
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

def BatchComp_AccumulateOp : BatchComp_Op<"accumulate", [Pure]> {
  let summary = "Accumulate tile results (K-dimension reduction)";
  let description = [{
    Accumulates partial results from different K-iterations into final output.

    For matmul C = A × B with K-dimension tiling:
    - K iteration 0: C_partial0 = A[:, 0:16] × B[0:16, :]
    - K iteration 1: C_partial1 = A[:, 16:32] × B[16:32, :]
    - Final: C = C_partial0 + C_partial1 + ...

    Example:
    ```mlir
    %C_final = batch_comp.accumulate %C_partial0, %C_partial1
               : tensor<16x16xf32>, tensor<16x16xf32> -> tensor<16x16xf32>
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Batch Management Operations
//===----------------------------------------------------------------------===//

def BatchComp_CreateBatchOp : BatchComp_Op<"create_batch", [Pure]> {
  let summary = "Create a new tile batch";
  let description = [{
    Creates an empty batch with hardware-specific capacity.

    Hardware configuration is specified via attributes:
    - batch_size: Number of tiles per batch
    - must_fill: Whether batch must be completely filled

    Example:
    ```mlir
    %batch = batch_comp.create_batch {
      batch_size = 16 : i32,
      hardware = "npu"
    } : !batch_comp.batch
    ```
  }];

  let arguments = (ins
    I32Attr:$batch_size,
    StrAttr:$hardware,
    DefaultValuedAttr<BoolAttr, "false">:$must_fill
  );
  let results = (outs BatchComp_Batch:$batch);

  let assemblyFormat = "attr-dict `:` type($batch)";
}

def BatchComp_AddToBatchOp : BatchComp_Op<"add_to_batch"> {
  let summary = "Add a tile to a batch";
  let description = [{
    Adds a tile computation to the batch at the specified slot.

    Attributes:
    - slot: Position in batch (0 to batch_size-1)
    - tile_id: Unique tile identifier
    - matrix_id: Which matmul this tile belongs to
    - Other metadata for tile location and size

    Example:
    ```mlir
    %batch1 = batch_comp.add_to_batch %batch0, %tile_a, %tile_b {
      slot = 0 : i32,
      tile_id = 0 : i32,
      matrix_id = 0 : i32,
      k_iteration = 0 : i64
    } : !batch_comp.batch, tensor<16x16xf32>, tensor<16x16xf32>
    ```
  }];

  let arguments = (ins
    BatchComp_Batch:$batch,
    AnyRankedTensor:$tile_a,
    AnyRankedTensor:$tile_b,
    I32Attr:$slot,
    I32Attr:$tile_id,
    I32Attr:$matrix_id,
    I64Attr:$k_iteration,
    I64Attr:$m_offset,
    I64Attr:$n_offset,
    I64Attr:$actual_m,
    I64Attr:$actual_n,
    I64Attr:$actual_k
  );
  let results = (outs BatchComp_Batch:$updated_batch);

  let assemblyFormat = [{
    $batch `,` $tile_a `,` $tile_b attr-dict
    `:` type($batch) `,` type($tile_a) `,` type($tile_b)
  }];
}

def BatchComp_ExecuteBatchOp : BatchComp_Op<"execute_batch"> {
  let summary = "Execute a batch of tiles on hardware";
  let description = [{
    Submits the batch for parallel execution on the target hardware.

    The actual execution strategy depends on hardware:
    - NPU: All 16 tiles execute in parallel on 16 cores
    - GPU: Tiles map to warp threads (32 threads)
    - TPU: Tiles feed into systolic array

    Example:
    ```mlir
    %result = batch_comp.execute_batch %batch {
      operation = "matmul",
      hardware = "npu"
    } : !batch_comp.batch -> !batch_comp.batch
    ```
  }];

  let arguments = (ins
    BatchComp_Batch:$batch,
    StrAttr:$operation,
    StrAttr:$hardware
  );
  let results = (outs BatchComp_Batch:$result);

  let assemblyFormat = [{
    $batch attr-dict `:` type($batch) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// High-Level Scheduling Operations
//===----------------------------------------------------------------------===//

def BatchComp_GenerateTilesOp : BatchComp_Op<"generate_tiles", [Pure]> {
  let summary = "Generate tiles for a matmul operation";
  let description = [{
    Decomposes a matmul into tiles based on tile size.

    This operation performs slicing and padding as needed for edge tiles.

    Example:
    ```mlir
    %tileset = batch_comp.generate_tiles %A, %B {
      tile_size = [16, 16, 16],  // M, N, K
      matrix_id = 0 : i32
    } : tensor<100x200xf32>, tensor<200x300xf32> -> !batch_comp.tileset
    ```
  }];

  let arguments = (ins
    AnyRankedTensor:$lhs,
    AnyRankedTensor:$rhs,
    I64ArrayAttr:$tile_size,  // [M, N, K]
    I32Attr:$matrix_id
  );
  let results = (outs BatchComp_TileSet:$tiles);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict
    `:` type($lhs) `,` type($rhs) `->` type($tiles)
  }];
}

def BatchComp_ScheduleBatchesOp : BatchComp_Op<"schedule_batches", [Pure]> {
  let summary = "Optimize tile-to-batch assignment";
  let description = [{
    Performs batch composition optimization to assign tiles to batches.

    This is where the core optimization happens. It can use different algorithms:
    - "greedy": Fast O(n) assignment
    - "improved_greedy": O(n log n) with grouping
    - "cpsat": Global optimization (NP with time limit)

    The algorithm is selected based on problem size and hardware config.

    Example:
    ```mlir
    %schedule = batch_comp.schedule_batches %tileset {
      hardware = "npu",
      batch_size = 16 : i32,
      algorithm = "auto"
    } : !batch_comp.tileset -> !batch_comp.schedule
    ```
  }];

  let arguments = (ins
    BatchComp_TileSet:$tiles,
    StrAttr:$hardware,
    I32Attr:$batch_size,
    DefaultValuedAttr<StrAttr, "\"auto\"">:$algorithm
  );
  let results = (outs BatchComp_BatchSchedule:$schedule);

  let assemblyFormat = "attr-dict $tiles `:` type($tiles) `->` type($schedule)";
}

def BatchComp_ExecuteScheduleOp : BatchComp_Op<"execute_schedule"> {
  let summary = "Execute the complete batch schedule";
  let description = [{
    Executes all batches in the schedule and assembles the final result.

    This operation:
    1. Executes each batch sequentially (batches are ordered by dependencies)
    2. Accumulates K-dimension partial results
    3. Assembles tiles back into the output tensor

    Example:
    ```mlir
    %C = batch_comp.execute_schedule %schedule {
      hardware = "npu"
    } : !batch_comp.schedule -> tensor<100x300xf32>
    ```
  }];

  let arguments = (ins
    BatchComp_BatchSchedule:$schedule,
    StrAttr:$hardware
  );
  let results = (outs AnyRankedTensor:$result);

  let assemblyFormat = [{
    $schedule attr-dict `:` type($schedule) `->` type($result)
  }];
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def BatchComp_NOPTileOp : BatchComp_Op<"nop_tile", [Pure]> {
  let summary = "Create a no-operation tile for padding batches";
  let description = [{
    Creates a dummy tile that performs no computation.
    Used when batch must be filled but fewer real tiles are available.

    Example:
    ```mlir
    %nop = batch_comp.nop_tile : tensor<16x16xf32>
    ```
  }];

  let results = (outs AnyRankedTensor:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
}

#endif // BATCHCOMP_OPS
