//===- NPUOps.td - NPU Dialect Operations ----------------*- tablegen -*-===//
//
// NPU Dialect for custom NPU chip with 16-way parallel tile execution
//
//===----------------------------------------------------------------------===//

#ifndef NPU_OPS
#define NPU_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// NPU Dialect
//===----------------------------------------------------------------------===//

def NPU_Dialect : Dialect {
  let name = "npu";
  let summary = "Operations for NPU hardware with 16-way parallel execution";
  let description = [{
    This dialect provides operations for custom NPU chip:
    - 16-way parallel compute cores
    - Variable-length tile support (up to 16×16)
    - Fixed batch size of 16 tiles
    - K-dimension reduction support
  }];
  let cppNamespace = "::mlir::npu";
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// NPU Types
//===----------------------------------------------------------------------===//

class NPU_Type<string name, string typeMnemonic> : TypeDef<NPU_Dialect, name> {
  let mnemonic = typeMnemonic;
}

def NPU_TileBatch : NPU_Type<"TileBatch", "batch"> {
  let summary = "NPU tile batch type";
  let description = [{
    Represents a batch of up to 16 tiles to be executed in parallel.
  }];
}

def NPU_BatchResult : NPU_Type<"BatchResult", "result"> {
  let summary = "NPU batch result type";
  let description = [{
    Represents the result of a batch execution containing 16 tile results.
  }];
}

//===----------------------------------------------------------------------===//
// NPU Operations
//===----------------------------------------------------------------------===//

class NPU_Op<string mnemonic, list<Trait> traits = []> :
  Op<NPU_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Batch Management Operations
//===----------------------------------------------------------------------===//

def NPU_CreateBatchOp : NPU_Op<"create_batch", [Pure]> {
  let summary = "Create a tile batch for NPU execution";
  let description = [{
    Creates an empty batch that can hold up to 16 tiles.

    Example:
    ```mlir
    %batch = npu.create_batch : !npu.batch
    ```
  }];

  let results = (outs NPU_TileBatch:$batch);
  let assemblyFormat = "attr-dict `:` type($batch)";
}

def NPU_AddTileToBatchOp : NPU_Op<"add_to_batch"> {
  let summary = "Add a tile pair (A, B) to the execution batch";
  let description = [{
    Adds a tile descriptor to the batch at the specified slot (0-15).
    Records the tile's source, size, offsets, and matrix ID.

    For matmul: C[m,n] += A[m,k] × B[k,n]
    - tile_a: A matrix tile [actual_m, actual_k]
    - tile_b: B matrix tile [actual_k, actual_n]

    Example:
    ```mlir
    %batch1 = npu.add_to_batch %batch0, %tile_a, %tile_b {
      slot = 0 : i32,
      matrix_id = 0 : i32,
      m_offset = 0 : i64,
      n_offset = 0 : i64,
      k_offset = 0 : i64,
      actual_m = 16 : i64,
      actual_n = 16 : i64,
      actual_k = 16 : i64
    } : !npu.batch, tensor<16x16xf32>, tensor<16x16xf32>
    ```
  }];

  let arguments = (ins
    NPU_TileBatch:$batch,
    AnyRankedTensor:$tile_a,
    AnyRankedTensor:$tile_b,
    I32Attr:$slot,
    I32Attr:$matrix_id,
    I64Attr:$m_offset,
    I64Attr:$n_offset,
    I64Attr:$k_offset,
    I64Attr:$actual_m,
    I64Attr:$actual_n,
    I64Attr:$actual_k
  );
  let results = (outs NPU_TileBatch:$updated_batch);

  let assemblyFormat = [{
    $batch `,` $tile_a `,` $tile_b attr-dict
    `:` type($batch) `,` type($tile_a) `,` type($tile_b)
  }];
}

//===----------------------------------------------------------------------===//
// Execution Operations
//===----------------------------------------------------------------------===//

def NPU_ExecuteBatchOp : NPU_Op<"execute_batch"> {
  let summary = "Execute a batch of tiles on NPU";
  let description = [{
    Submits the batch to NPU for parallel execution.
    All 16 tiles execute simultaneously on 16 compute cores.

    For matmul: Each tile computes partial_C[m,n] = A[m,k] × B[k,n]
    Note: This is a partial result that needs to be accumulated across K.

    Example:
    ```mlir
    %result = npu.execute_batch %batch {operation = "matmul"}
              : !npu.batch -> !npu.result
    ```
  }];

  let arguments = (ins
    NPU_TileBatch:$batch,
    StrAttr:$operation
  );
  let results = (outs NPU_BatchResult:$result);

  let assemblyFormat = [{
    $batch attr-dict `:` type($batch) `->` type($result)
  }];
}

def NPU_ExtractResultOp : NPU_Op<"extract_result", [Pure]> {
  let summary = "Extract a tile result from batch execution";
  let description = [{
    Extracts the result of a specific tile from the batch result.

    Example:
    ```mlir
    %tile_result = npu.extract_result %result {slot = 0 : i32}
                   : !npu.result -> tensor<16x16xf32>
    ```
  }];

  let arguments = (ins
    NPU_BatchResult:$batch_result,
    I32Attr:$slot
  );
  let results = (outs AnyRankedTensor:$tile_result);

  let assemblyFormat = [{
    $batch_result attr-dict `:` type($batch_result) `->` type($tile_result)
  }];
}

//===----------------------------------------------------------------------===//
// Utility Operations
//===----------------------------------------------------------------------===//

def NPU_NOPTileOp : NPU_Op<"nop_tile", [Pure]> {
  let summary = "Create a NOP (no-operation) tile";
  let description = [{
    Creates a dummy tile that occupies a slot but performs no computation.
    Used to fill the batch when fewer than 16 real tiles are available.

    Example:
    ```mlir
    %nop_a = npu.nop_tile : tensor<16x16xf32>
    %nop_b = npu.nop_tile : tensor<16x16xf32>
    ```
  }];

  let results = (outs AnyRankedTensor:$tile);
  let assemblyFormat = "attr-dict `:` type($tile)";
}

#endif // NPU_OPS
