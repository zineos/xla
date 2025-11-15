//===- NPUPasses.h - NPU Transformation Passes -------------------*- C++ -*-===//

#ifndef NPU_NPUPASSES_H
#define NPU_NPUPASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace npu {

#define GEN_PASS_DECL
#include "NPU/NPUPasses.h.inc"

/// Create NPU Tile Scheduler Pass
std::unique_ptr<Pass> createNPUTileSchedulerPass();

/// Register all NPU passes
#define GEN_PASS_REGISTRATION
#include "NPU/NPUPasses.h.inc"

} // namespace npu
} // namespace mlir

#endif // NPU_NPUPASSES_H
