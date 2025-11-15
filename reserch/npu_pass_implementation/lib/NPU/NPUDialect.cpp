//===- NPUDialect.cpp - NPU Dialect Definition ----------------------------===//

#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"

// Include generated code
#define GET_TYPEDEF_CLASSES
#include "NPU/NPUTypes.cpp.inc"

#include "NPU/NPUDialect.cpp.inc"

namespace mlir {
namespace npu {

void NPUDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "NPU/NPUOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "NPU/NPUTypes.cpp.inc"
      >();
}

} // namespace npu
} // namespace mlir
