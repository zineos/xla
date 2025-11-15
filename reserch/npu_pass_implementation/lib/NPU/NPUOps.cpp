//===- NPUOps.cpp - NPU Dialect Operations -------------------------------===//

#include "NPU/NPUOps.h"
#include "NPU/NPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "NPU/NPUOps.cpp.inc"

namespace mlir {
namespace npu {

// Operation implementations can be added here if needed

} // namespace npu
} // namespace mlir
