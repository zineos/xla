//===- NPUOps.h - NPU Dialect Operations -------------------------*- C++ -*-===//

#ifndef NPU_NPUOPS_H
#define NPU_NPUOPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "NPU/NPUTypes.h.inc"

#define GET_OP_CLASSES
#include "NPU/NPUOps.h.inc"

#endif // NPU_NPUOPS_H
