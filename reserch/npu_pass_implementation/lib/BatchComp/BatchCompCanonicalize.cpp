//===- BatchCompCanonicalize.cpp - Canonicalization Patterns -------------===//
//
// Applies canonicalization and optimization patterns to BatchComp IR
//
//===----------------------------------------------------------------------===//

#include "BatchComp/BatchCompPasses.h"
#include "BatchComp/BatchCompOps.h"
#include "BatchComp/BatchCompDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "batch-comp-canonicalize"

namespace mlir {
namespace batch_comp {

#define GEN_PASS_DEF_BATCHCOMPCANONICALIZEPASS
#include "BatchComp/BatchCompPasses.h.inc"

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

/// Fold consecutive accumulate operations
///
/// Transforms:
/// %tmp = batch_comp.accumulate %a, %b
/// %result = batch_comp.accumulate %tmp, %c
///
/// To:
/// %result = batch_comp.accumulate %a, %b, %c  (if multi-operand supported)
///
/// Or keep as-is for now (accumulate is associative)
class FoldConsecutiveAccumulate : public OpRewritePattern<AccumulateOp> {
public:
  using OpRewritePattern<AccumulateOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AccumulateOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if lhs is also an accumulate
    auto lhsAccumOp = op.getLhs().getDefiningOp<AccumulateOp>();
    if (!lhsAccumOp) {
      return failure();
    }

    // Check if lhs accumulate has only one use
    if (!lhsAccumOp.getResult().hasOneUse()) {
      return failure();
    }

    LLVM_DEBUG(llvm::dbgs() << "Folding consecutive accumulate operations\n");

    // For now, accumulate is binary, so we keep the chain
    // In a future extension, we could support variadic accumulate:
    // %result = batch_comp.accumulate %a, %b, %c

    // No optimization for now - accumulate chain is fine
    return failure();
  }
};

/// Remove redundant padding
///
/// Transforms:
/// %padded = batch_comp.pad %input to [16, 16]
/// (where %input is already [16, 16])
///
/// To:
/// (use %input directly)
class RemoveRedundantPad : public OpRewritePattern<PadOp> {
public:
  using OpRewritePattern<PadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(PadOp op,
                                 PatternRewriter &rewriter) const override {
    auto inputType = op.getInput().getType().cast<RankedTensorType>();
    auto resultType = op.getResult().getType().cast<RankedTensorType>();

    // Check if input and result have the same shape
    if (inputType.getShape() == resultType.getShape()) {
      LLVM_DEBUG(llvm::dbgs() << "Removing redundant pad operation\n");

      // Replace pad with its input
      rewriter.replaceOp(op, op.getInput());
      return success();
    }

    return failure();
  }
};

/// Merge slice operations with compatible ranges
///
/// Transforms:
/// %slice1 = batch_comp.slice %A[0, 0] [16, 16]
/// %slice2 = batch_comp.slice %A[0, 16] [16, 16]
///
/// To:
/// %combined = batch_comp.slice %A[0, 0] [16, 32]
/// (then extract slice1 and slice2 if needed)
///
/// This is useful when adjacent slices can be loaded together
class MergeAdjacentSlices : public OpRewritePattern<SliceOp> {
public:
  using OpRewritePattern<SliceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(SliceOp op,
                                 PatternRewriter &rewriter) const override {
    // This is a more complex optimization that requires:
    // - Finding other slice ops from the same source
    // - Checking if they are adjacent
    // - Merging them into a larger slice
    // - Potentially adding subsequent slice ops to extract the parts

    // For now, skip this optimization
    return failure();
  }
};

/// Eliminate dead batches
///
/// Removes create_batch operations that are never used or executed
class EliminateDeadBatches : public OpRewritePattern<CreateBatchOp> {
public:
  using OpRewritePattern<CreateBatchOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CreateBatchOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if the batch is ever used
    if (op.getBatch().use_empty()) {
      LLVM_DEBUG(llvm::dbgs() << "Eliminating dead batch\n");
      rewriter.eraseOp(op);
      return success();
    }

    // Check if the batch is only used in add_to_batch but never executed
    bool hasExecute = false;
    for (auto user : op.getBatch().getUsers()) {
      if (isa<ExecuteBatchOp>(user)) {
        hasExecute = true;
        break;
      }
      // Also check transitively through add_to_batch chains
      if (auto addOp = dyn_cast<AddToBatchOp>(user)) {
        for (auto batchUser : addOp.getUpdatedBatch().getUsers()) {
          if (isa<ExecuteBatchOp>(batchUser)) {
            hasExecute = true;
            break;
          }
        }
      }
    }

    if (!hasExecute) {
      LLVM_DEBUG(llvm::dbgs() << "Eliminating batch without execution\n");
      // Would need to erase the whole chain - skip for now
      return failure();
    }

    return failure();
  }
};

/// Fold NOP tiles
///
/// Removes unnecessary NOP tiles from batches if the batch doesn't require filling
class FoldNOPTiles : public OpRewritePattern<NOPTileOp> {
public:
  using OpRewritePattern<NOPTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(NOPTileOp op,
                                 PatternRewriter &rewriter) const override {
    // Check if this NOP tile is used in a batch that doesn't require filling
    // This would need analysis of the add_to_batch and create_batch ops

    // For now, keep NOP tiles as they are necessary for must_fill batches
    return failure();
  }
};

//===----------------------------------------------------------------------===//
// BatchCompCanonicalizePass
//===----------------------------------------------------------------------===//

class BatchCompCanonicalizePass
    : public impl::BatchCompCanonicalizePassBase<BatchCompCanonicalizePass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();

    LLVM_DEBUG(llvm::dbgs() << "Running BatchComp Canonicalize Pass\n");

    // Set up rewrite patterns
    RewritePatternSet patterns(context);
    patterns.add<FoldConsecutiveAccumulate>(context);
    patterns.add<RemoveRedundantPad>(context);
    patterns.add<MergeAdjacentSlices>(context);
    patterns.add<EliminateDeadBatches>(context);
    patterns.add<FoldNOPTiles>(context);

    // Apply patterns greedily
    GreedyRewriteConfig config;
    config.useTopDownTraversal = true;
    config.maxIterations = 10;

    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns), config))) {
      LLVM_DEBUG(llvm::dbgs() << "Some canonicalization patterns did not converge\n");
      // Don't signal failure - this is not necessarily an error
    }

    LLVM_DEBUG(llvm::dbgs() << "BatchComp Canonicalize Pass complete\n");
  }
};

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createBatchCompCanonicalizePass() {
  return std::make_unique<BatchCompCanonicalizePass>();
}

} // namespace batch_comp
} // namespace mlir
