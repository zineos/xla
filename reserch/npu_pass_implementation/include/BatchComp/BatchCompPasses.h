//===- BatchCompPasses.h - Batch Composition Passes -------------*- C++ -*-===//
//
// Header for Batch Composition Optimization Passes
//
//===----------------------------------------------------------------------===//

#ifndef BATCHCOMP_PASSES_H
#define BATCHCOMP_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace batch_comp {

//===----------------------------------------------------------------------===//
// Pass Creation Functions
//===----------------------------------------------------------------------===//

/// Create the BatchComp Tile Scheduler Pass.
///
/// This pass transforms TOSA/Linalg matmul operations into optimized
/// BatchComp dialect operations with intelligent tile-to-batch scheduling.
///
/// Example:
/// ```cpp
/// pm.addPass(batch_comp::createBatchCompTileSchedulerPass());
/// ```
std::unique_ptr<Pass> createBatchCompTileSchedulerPass();

/// Create the BatchComp Lowering Pass.
///
/// This pass lowers high-level BatchComp IR (generate_tiles, schedule_batches,
/// execute_schedule) to low-level BatchComp IR (slice, pad, create_batch,
/// add_to_batch, execute_batch, accumulate).
///
/// Example:
/// ```cpp
/// pm.addPass(batch_comp::createBatchCompLoweringPass());
/// ```
std::unique_ptr<Pass> createBatchCompLoweringPass();

/// Create the BatchComp to NPU Pass.
///
/// This pass translates BatchComp low-level operations to NPU dialect
/// operations for final code generation.
///
/// Example:
/// ```cpp
/// pm.addPass(batch_comp::createBatchCompToNPUPass());
/// ```
std::unique_ptr<Pass> createBatchCompToNPUPass();

/// Create the BatchComp Canonicalization Pass.
///
/// This pass applies canonicalization patterns to simplify BatchComp IR.
///
/// Example:
/// ```cpp
/// pm.addPass(batch_comp::createBatchCompCanonicalizePass());
/// ```
std::unique_ptr<Pass> createBatchCompCanonicalizePass();

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "BatchComp/BatchCompPasses.h.inc"

} // namespace batch_comp
} // namespace mlir

#endif // BATCHCOMP_PASSES_H
