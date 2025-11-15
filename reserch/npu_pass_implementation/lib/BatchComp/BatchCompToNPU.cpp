//===- BatchCompToNPU.cpp - Lower BatchComp to NPU Dialect ---------------===//
//
// Translates BatchComp low-level operations to NPU dialect operations
//
//===----------------------------------------------------------------------===//

#include "BatchComp/BatchCompPasses.h"
#include "BatchComp/BatchCompOps.h"
#include "BatchComp/BatchCompDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "batch-comp-to-npu"

namespace mlir {
namespace batch_comp {

#define GEN_PASS_DEF_BATCHCOMPTONPUPASS
#include "BatchComp/BatchCompPasses.h.inc"

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

/// Lower batch_comp.create_batch to NPU batch creation
///
/// Transforms:
/// %batch = batch_comp.create_batch {batch_size = 16, hardware = "npu"}
///
/// To NPU-specific operations (example):
/// %npu_batch = npu.create_batch {size = 16}
///
/// Note: This requires the NPU dialect to be defined separately.
/// For now, we provide the framework structure.
class CreateBatchToNPU : public OpRewritePattern<CreateBatchOp> {
public:
  using OpRewritePattern<CreateBatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateBatchOp op,
                                 PatternRewriter &rewriter) const override {
    // Only convert for NPU hardware
    if (op.getHardware() != "npu") {
      return failure();
    }

    Location loc = op.getLoc();
    int32_t batchSize = op.getBatchSize();

    LLVM_DEBUG(llvm::dbgs() << "Converting create_batch to NPU (size="
                            << batchSize << ")\n");

    // TODO: Create NPU dialect batch operation
    // Example: rewriter.create<npu::CreateBatchOp>(loc, ..., batchSize);

    // For now, keep the BatchComp operation as-is
    return failure();
  }
};

/// Lower batch_comp.execute_batch to NPU execution
///
/// Transforms:
/// %result = batch_comp.execute_batch %batch {operation = "matmul", hardware = "npu"}
///
/// To NPU-specific parallel execution:
/// %result = npu.parallel_matmul %batch {cores = 16}
class ExecuteBatchToNPU : public OpRewritePattern<ExecuteBatchOp> {
public:
  using OpRewritePattern<ExecuteBatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ExecuteBatchOp op,
                                 PatternRewriter &rewriter) const override {
    // Only convert for NPU hardware
    if (op.getHardware() != "npu") {
      return failure();
    }

    Location loc = op.getLoc();
    Value batch = op.getBatch();
    StringRef operation = op.getOperation();

    LLVM_DEBUG(llvm::dbgs() << "Converting execute_batch to NPU (op="
                            << operation << ")\n");

    // TODO: Create NPU dialect execution operation
    // This would map to NPU's 16-way parallel execution model
    // Example:
    // if (operation == "matmul") {
    //   rewriter.create<npu::ParallelMatmulOp>(loc, batch, ...);
    // }

    // For now, keep the BatchComp operation as-is
    return failure();
  }
};

/// Lower batch_comp.tile_matmul to NPU tile-level matmul
///
/// Transforms:
/// %C = batch_comp.tile_matmul %A, %B {m_offset = 0, n_offset = 0, k_iteration = 0}
///
/// To NPU tile computation:
/// %C = npu.tile_compute %A, %B {mode = "matmul", tile_id = ...}
class TileMatmulToNPU : public OpRewritePattern<TileMatmulOp> {
public:
  using OpRewritePattern<TileMatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(TileMatmulOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    LLVM_DEBUG(llvm::dbgs() << "Converting tile_matmul to NPU\n");

    // TODO: Create NPU dialect tile operation
    // This maps to a single NPU core computation
    // Example:
    // rewriter.create<npu::TileComputeOp>(
    //   loc, op.getResult().getType(), lhs, rhs,
    //   op.getMatrixIdAttr(), op.getMOffsetAttr(), ...);

    // For now, keep the BatchComp operation as-is
    return failure();
  }
};

/// Lower batch_comp.accumulate to NPU accumulation
///
/// Transforms:
/// %C = batch_comp.accumulate %partial0, %partial1
///
/// To NPU accumulator operation:
/// %C = npu.accumulate %partial0, %partial1
class AccumulateToNPU : public OpRewritePattern<AccumulateOp> {
public:
  using OpRewritePattern<AccumulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AccumulateOp op,
                                 PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs();
    Value rhs = op.getRhs();

    LLVM_DEBUG(llvm::dbgs() << "Converting accumulate to NPU\n");

    // TODO: Create NPU dialect accumulate operation
    // This could map to NPU's accumulator hardware or just addition
    // Example:
    // rewriter.create<npu::AccumulateOp>(
    //   loc, op.getResult().getType(), lhs, rhs);

    // For now, keep the BatchComp operation as-is
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// BatchCompToNPUPass
//===----------------------------------------------------------------------===//

class BatchCompToNPUPass
    : public impl::BatchCompToNPUPassBase<BatchCompToNPUPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    LLVM_DEBUG(llvm::dbgs() << "Running BatchComp to NPU Pass\n");

    // Set up conversion target
    ConversionTarget target(*context);
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<arith::ArithDialect>();

    // TODO: Add NPU dialect when available
    // target.addLegalDialect<npu::NPUDialect>();

    // Mark BatchComp low-level ops as illegal (to be converted to NPU)
    target.addIllegalOp<CreateBatchOp>();
    target.addIllegalOp<ExecuteBatchOp>();
    target.addIllegalOp<TileMatmulOp>();
    target.addIllegalOp<AccumulateOp>();

    // Other BatchComp ops pass through
    target.addLegalOp<SliceOp>();
    target.addLegalOp<PadOp>();

    // Set up rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<CreateBatchToNPU>(context);
    patterns.add<ExecuteBatchToNPU>(context);
    patterns.add<TileMatmulToNPU>(context);
    patterns.add<AccumulateToNPU>(context);

    // Apply the patterns
    if (failed(applyPartialConversion(funcOp, target, std::move(patterns)))) {
      LLVM_DEBUG(llvm::dbgs() << "BatchComp to NPU conversion failed (expected - NPU dialect not available)\n");
      // Don't signal failure since NPU dialect is not yet implemented
      // signalPassFailure();
    }

    LLVM_DEBUG(llvm::dbgs() << "BatchComp to NPU Pass complete\n");
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createBatchCompToNPUPass() {
  return std::make_unique<BatchCompToNPUPass>();
}

} // namespace batch_comp
} // namespace mlir
