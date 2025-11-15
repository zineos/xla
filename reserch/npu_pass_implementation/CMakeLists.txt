# CMakeLists.txt for NPU Tile Scheduler Pass

cmake_minimum_required(VERSION 3.20)
project(npu-tile-scheduler)

# Find MLIR
find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

# Generate NPU Dialect files
set(LLVM_TARGET_DEFINITIONS include/NPU/NPUOps.td)
mlir_tablegen(NPUOps.h.inc -gen-op-decls)
mlir_tablegen(NPUOps.cpp.inc -gen-op-defs)
mlir_tablegen(NPUDialect.h.inc -gen-dialect-decls)
mlir_tablegen(NPUDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(NPUTypes.h.inc -gen-typedef-decls)
mlir_tablegen(NPUTypes.cpp.inc -gen-typedef-defs)
add_public_tablegen_target(MLIRNPUOpsIncGen)

# Generate NPU Pass files
set(LLVM_TARGET_DEFINITIONS include/NPU/NPUPasses.td)
mlir_tablegen(NPUPasses.h.inc -gen-pass-decls -name NPU)
mlir_tablegen(NPUPasses.capi.h.inc -gen-pass-capi-header --prefix NPU)
mlir_tablegen(NPUPasses.capi.cpp.inc -gen-pass-capi-impl --prefix NPU)
add_public_tablegen_target(MLIRNPUPassIncGen)

# NPU Dialect library
add_mlir_dialect_library(MLIRNPUDialect
  lib/NPU/NPUDialect.cpp
  lib/NPU/NPUOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU

  DEPENDS
  MLIRNPUOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)

# NPU Pass library
add_mlir_library(MLIRNPUTransforms
  lib/NPU/NPUTileSchedulerPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/NPU

  DEPENDS
  MLIRNPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransformUtils
  MLIRTosaDialect
  MLIRTensorDialect
  MLIRArithDialect
  # MLIRSCFDialect removed - K-loop is fully unrolled
  MLIRLinalgDialect
  MLIRNPUDialect
)

# ============================================================================
# BatchComp Dialect (New Unified Framework)
# ============================================================================

# Generate BatchComp Dialect files
set(LLVM_TARGET_DEFINITIONS include/BatchComp/BatchCompOps.td)
mlir_tablegen(include/BatchComp/BatchCompOps.h.inc -gen-op-decls)
mlir_tablegen(include/BatchComp/BatchCompOps.cpp.inc -gen-op-defs)
mlir_tablegen(include/BatchComp/BatchCompDialect.h.inc -gen-dialect-decls -dialect=batch_comp)
mlir_tablegen(include/BatchComp/BatchCompDialect.cpp.inc -gen-dialect-defs -dialect=batch_comp)
mlir_tablegen(include/BatchComp/BatchCompTypes.h.inc -gen-typedef-decls -typedefs-dialect=batch_comp)
mlir_tablegen(include/BatchComp/BatchCompTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=batch_comp)
add_public_tablegen_target(MLIRBatchCompOpsIncGen)

# Generate BatchComp Pass files
set(LLVM_TARGET_DEFINITIONS include/BatchComp/BatchCompPasses.td)
mlir_tablegen(include/BatchComp/BatchCompPasses.h.inc -gen-pass-decls -name BatchComp)
add_public_tablegen_target(MLIRBatchCompPassIncGen)

# BatchComp Dialect library
# Note: Requires BatchCompDialect.cpp, BatchCompOps.cpp, BatchCompTypes.cpp
# These files would contain the C++ implementation of the dialect
# add_mlir_dialect_library(MLIRBatchCompDialect
#   lib/BatchComp/BatchCompDialect.cpp
#   lib/BatchComp/BatchCompOps.cpp
#   lib/BatchComp/BatchCompTypes.cpp
#
#   ADDITIONAL_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}/include/BatchComp
#
#   DEPENDS
#   MLIRBatchCompOpsIncGen
#
#   LINK_LIBS PUBLIC
#   MLIRIR
#   MLIRInferTypeOpInterface
#   MLIRSideEffectInterfaces
# )

# Optional: Python Integration
option(ENABLE_PYTHON_INTEGRATION "Enable Python optimizer integration via pybind11" ON)
if(ENABLE_PYTHON_INTEGRATION)
  find_package(Python3 COMPONENTS Interpreter Development)
  if(Python3_FOUND)
    find_package(pybind11 CONFIG)
    if(pybind11_FOUND)
      message(STATUS "Found pybind11 - Python integration enabled")
      add_compile_definitions(ENABLE_PYBIND11)
    else()
      message(WARNING "pybind11 not found - Python integration disabled")
    endif()
  endif()
endif()

# BatchComp Pass library
# Commented out until dialect implementation files are created
# add_mlir_library(MLIRBatchCompPasses
#   lib/BatchComp/BatchCompTileScheduler.cpp
#   lib/BatchComp/BatchCompLowering.cpp
#   lib/BatchComp/BatchCompToNPU.cpp
#   lib/BatchComp/BatchCompCanonicalize.cpp
#   lib/BatchComp/PythonIntegration.cpp
#
#   ADDITIONAL_HEADER_DIRS
#   ${PROJECT_SOURCE_DIR}/include/BatchComp
#
#   DEPENDS
#   MLIRBatchCompPassIncGen
#   MLIRBatchCompDialect
#
#   LINK_LIBS PUBLIC
#   MLIRBatchCompDialect
#   MLIRPass
#   MLIRTransforms
#   MLIRTosaDialect
#   MLIRLinalgDialect
#   MLIRFuncDialect
#   MLIRTensorDialect
#   MLIRArithDialect
#   MLIRSupport
#   MLIRTransformUtils
# )
#
# if(ENABLE_PYBIND11 AND pybind11_FOUND)
#   target_link_libraries(MLIRBatchCompPasses PRIVATE pybind11::embed)
# endif()

# ============================================================================
# Test Targets
# ============================================================================

# Test target for NPU dialect (legacy)
add_custom_target(check-npu
  COMMAND ${CMAKE_COMMAND} -E echo "Running NPU Pass tests..."
  COMMAND mlir-opt ${PROJECT_SOURCE_DIR}/test/npu_tile_scheduler_test.mlir -npu-tile-scheduler
  DEPENDS MLIRNPUTransforms
)

# Test target for BatchComp dialect (new)
# Uncomment when MLIRBatchCompPasses is built
# add_custom_target(check-batchcomp
#   COMMAND ${CMAKE_COMMAND} -E echo "Running BatchComp Pass tests..."
#   COMMAND mlir-opt ${PROJECT_SOURCE_DIR}/test/batch_comp_test.mlir \
#     --batch-comp-tile-scheduler="hardware=npu tile-size=16,16,16 algorithm=cpsat"
#   DEPENDS MLIRBatchCompPasses
# )

# ============================================================================
# Build Summary
# ============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Build Configuration Summary")
message(STATUS "========================================")
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "MLIR Dir: ${MLIR_DIR}")
message(STATUS "Python Integration: ${ENABLE_PYTHON_INTEGRATION}")
if(ENABLE_PYBIND11 AND pybind11_FOUND)
  message(STATUS "  pybind11: ENABLED")
else()
  message(STATUS "  pybind11: DISABLED")
endif()
message(STATUS "========================================")
message(STATUS "")
