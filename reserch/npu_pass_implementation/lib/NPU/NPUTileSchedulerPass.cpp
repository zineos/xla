//===- NPUTileSchedulerPass.cpp - NPU Tile Scheduler Pass ----------------===//
//
// Implements the NPU Tile Scheduler Pass for tosa.matmul operations
//
//===----------------------------------------------------------------------===//

#include "NPU/NPUDialect.h"
#include "NPU/NPUOps.h"
#include "NPU/NPUPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
// Note: SCF dialect removed - K-loop is fully unrolled
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"
#include <algorithm>
#include <vector>

#define DEBUG_TYPE "npu-tile-scheduler"

namespace mlir {
namespace npu {

#define GEN_PASS_DEF_NPUTILESCHEDULERPASS
#include "NPU/NPUPasses.h.inc"

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

struct TileCandidate {
  int matrixId;
  int64_t mOffset, nOffset, kOffset;
  int64_t actualM, actualN, actualK;
  float paddingRatio;
  Value tileA, tileB;

  TileCandidate() = default;
  TileCandidate(int mid, int64_t m, int64_t n, int64_t k, int64_t am,
                int64_t an, int64_t ak, Value a, Value b)
      : matrixId(mid), mOffset(m), nOffset(n), kOffset(k), actualM(am),
        actualN(an), actualK(ak), tileA(a), tileB(b) {
    // Calculate padding ratio for M×N tile
    // (K dimension doesn't affect per-tile padding as all tiles in same batch
    // have same K-slice)
    int64_t tileCapacity = 16 * 16; // M×N capacity
    int64_t actualElements = actualM * actualN;
    paddingRatio = 1.0f - static_cast<float>(actualElements) / tileCapacity;
  }

  bool operator<(const TileCandidate &other) const {
    return paddingRatio < other.paddingRatio; // Complete tiles first
  }
};

struct MatmulInfo {
  tosa::MatMulOp op;
  Value inputA, inputB, result;
  int64_t M, K, N;
  int matrixId;
};

//===----------------------------------------------------------------------===//
// NPU Tile Scheduler Pass
//===----------------------------------------------------------------------===//

class NPUTileSchedulerPass
    : public impl::NPUTileSchedulerPassBase<NPUTileSchedulerPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    OpBuilder builder(funcOp.getContext());

    // Step 1: Collect all tosa.matmul operations
    SmallVector<MatmulInfo> matmuls;
    int matrixId = 0;

    funcOp.walk([&](tosa::MatMulOp matmulOp) {
      auto aType = matmulOp.getA().getType().dyn_cast<RankedTensorType>();
      auto bType = matmulOp.getB().getType().dyn_cast<RankedTensorType>();
      auto cType =
          matmulOp.getResult().getType().dyn_cast<RankedTensorType>();

      if (!aType || !bType || !cType)
        return;

      ArrayRef<int64_t> aShape = aType.getShape();
      ArrayRef<int64_t> bShape = bType.getShape();

      // tosa.matmul: C[M,N] = A[M,K] × B[K,N]
      int64_t M = aShape[0];
      int64_t K = aShape[1];
      int64_t N = bShape[1];

      MatmulInfo info;
      info.op = matmulOp;
      info.inputA = matmulOp.getA();
      info.inputB = matmulOp.getB();
      info.result = matmulOp.getResult();
      info.M = M;
      info.K = K;
      info.N = N;
      info.matrixId = matrixId++;

      matmuls.push_back(info);
    });

    if (matmuls.empty())
      return;

    // Step 2: Transform each matmul
    for (const auto &info : matmuls) {
      transformMatmul(info, builder);
    }
  }

private:
  void transformMatmul(const MatmulInfo &info, OpBuilder &builder) {
    Location loc = info.op.getLoc();
    builder.setInsertionPoint(info.op);

    // Create constants
    Value c0_f32 = builder.create<arith::ConstantFloatOp>(
        loc, APFloat(0.0f), builder.getF32Type());

    // Initialize accumulator (output tensor filled with 0)
    auto cType = info.result.getType().cast<RankedTensorType>();
    Value cInit = builder.create<tensor::EmptyOp>(loc, cType, ValueRange{});
    Value accumulator =
        builder.create<linalg::FillOp>(loc, ValueRange{c0_f32}, ValueRange{cInit})
            .getResult(0);

    // Calculate number of K-iterations
    int64_t numKIterations = (info.K + tileSize - 1) / tileSize;

    // Unroll K-loop: generate separate code for each K-iteration
    for (int64_t kIter = 0; kIter < numKIterations; ++kIter) {
      int64_t kStart = kIter * tileSize;
      int64_t kEnd = std::min<int64_t>(kStart + tileSize, info.K);
      int64_t actualK = kEnd - kStart;

      // Create K-offset constant
      Value kOffset = builder.create<arith::ConstantIndexOp>(loc, kStart);
      Value kSizeVal = builder.create<arith::ConstantIndexOp>(loc, actualK);

      // Generate M×N tile candidates for this K-slice
      SmallVector<TileCandidate> candidates =
          generateTileCandidatesUnrolled(info, kOffset, kSizeVal, kStart, actualK, builder, loc);

      // Sort by padding ratio
      std::sort(candidates.begin(), candidates.end());

      // Select up to 16 best tiles
      unsigned numTiles = std::min<unsigned>(candidates.size(), maxTilesPerBatch);
      SmallVector<TileCandidate> selectedTiles(candidates.begin(),
                                               candidates.begin() + numTiles);

      // Fill with NOP if needed
      while (selectedTiles.size() < maxTilesPerBatch) {
        TileCandidate nop;
        nop.matrixId = -1; // Mark as NOP
        nop.paddingRatio = 1.0f;
        selectedTiles.push_back(nop);
      }

      // Generate batch execution and accumulate
      accumulator = executeBatch(selectedTiles, accumulator, info, builder, loc);
    }

    // Replace original matmul with the accumulated result
    info.op.replaceAllUsesWith(ValueRange{accumulator});
    info.op.erase();
  }

  // New unrolled version (no dynamic loop)
  SmallVector<TileCandidate> generateTileCandidatesUnrolled(
      const MatmulInfo &info, Value kOffset, Value kSizeVal,
      int64_t kStart, int64_t actualK, OpBuilder &builder, Location loc) {
    SmallVector<TileCandidate> candidates;

    // Generate all M×N tile positions
    for (int64_t m = 0; m < info.M; m += tileSize) {
      for (int64_t n = 0; n < info.N; n += tileSize) {
        int64_t actualM = std::min<int64_t>(tileSize, info.M - m);
        int64_t actualN = std::min<int64_t>(tileSize, info.N - n);

        // Extract tile from A: A[m:m+actualM, kStart:kStart+actualK]
        Value mOffset = builder.create<arith::ConstantIndexOp>(loc, m);
        Value nOffset = builder.create<arith::ConstantIndexOp>(loc, n);

        SmallVector<OpFoldResult> aOffsets = {mOffset, kOffset};
        SmallVector<OpFoldResult> aSizes = {builder.getIndexAttr(actualM),
                                            kSizeVal};
        SmallVector<OpFoldResult> aStrides = {builder.getIndexAttr(1),
                                              builder.getIndexAttr(1)};

        Value tileA = builder.create<tensor::ExtractSliceOp>(
            loc, info.inputA, aOffsets, aSizes, aStrides);

        // Extract tile from B: B[kStart:kStart+actualK, n:n+actualN]
        SmallVector<OpFoldResult> bOffsets = {kOffset, nOffset};
        SmallVector<OpFoldResult> bSizes = {kSizeVal,
                                            builder.getIndexAttr(actualN)};
        SmallVector<OpFoldResult> bStrides = {builder.getIndexAttr(1),
                                              builder.getIndexAttr(1)};

        Value tileB = builder.create<tensor::ExtractSliceOp>(
            loc, info.inputB, bOffsets, bSizes, bStrides);

        TileCandidate candidate(info.matrixId, m, n, kStart, actualM, actualN,
                                actualK, tileA, tileB);
        candidates.push_back(candidate);
      }
    }

    return candidates;
  }

  // Original version (kept for reference, not used after unrolling)
  SmallVector<TileCandidate> generateTileCandidates(const MatmulInfo &info,
                                                     Value kIdx,
                                                     Value kTileSize,
                                                     OpBuilder &builder,
                                                     Location loc,
                                                     int64_t kIteration) {
    SmallVector<TileCandidate> candidates;

    // Calculate actual K size for this iteration
    int64_t kStart = kIteration * tileSize;
    int64_t kEnd = std::min<int64_t>(kStart + tileSize, info.K);
    int64_t actualK = kEnd - kStart;

    // Generate all M×N tile positions
    for (int64_t m = 0; m < info.M; m += tileSize) {
      for (int64_t n = 0; n < info.N; n += tileSize) {
        int64_t actualM = std::min<int64_t>(tileSize, info.M - m);
        int64_t actualN = std::min<int64_t>(tileSize, info.N - n);

        // Extract tile from A: A[m:m+actualM, k:k+actualK]
        Value mOffset = builder.create<arith::ConstantIndexOp>(loc, m);
        Value nOffset = builder.create<arith::ConstantIndexOp>(loc, n);

        SmallVector<OpFoldResult> aOffsets = {mOffset, kIdx};
        SmallVector<OpFoldResult> aSizes = {
            builder.getIndexAttr(actualM), kTileSize};
        SmallVector<OpFoldResult> aStrides = {builder.getIndexAttr(1),
                                              builder.getIndexAttr(1)};

        Value tileA = builder.create<tensor::ExtractSliceOp>(
            loc, info.inputA, aOffsets, aSizes, aStrides);

        // Extract tile from B: B[k:k+actualK, n:n+actualN]
        SmallVector<OpFoldResult> bOffsets = {kIdx, nOffset};
        SmallVector<OpFoldResult> bSizes = {kTileSize,
                                            builder.getIndexAttr(actualN)};
        SmallVector<OpFoldResult> bStrides = {builder.getIndexAttr(1),
                                              builder.getIndexAttr(1)};

        Value tileB = builder.create<tensor::ExtractSliceOp>(
            loc, info.inputB, bOffsets, bSizes, bStrides);

        TileCandidate candidate(info.matrixId, m, n, kStart, actualM, actualN,
                                actualK, tileA, tileB);
        candidates.push_back(candidate);
      }
    }

    return candidates;
  }

  Value executeBatch(ArrayRef<TileCandidate> tiles, Value accumulator,
                     const MatmulInfo &info, OpBuilder &builder, Location loc) {
    // Create batch
    auto batchType = TileBatchType::get(builder.getContext());
    Value batch = builder.create<CreateBatchOp>(loc, batchType);

    // Add tiles to batch
    for (size_t i = 0; i < tiles.size(); ++i) {
      const TileCandidate &tile = tiles[i];

      if (tile.matrixId == -1) {
        // NOP tile
        auto nopType = RankedTensorType::get({16, 16}, builder.getF32Type());
        Value nopA = builder.create<NOPTileOp>(loc, nopType);
        Value nopB = builder.create<NOPTileOp>(loc, nopType);

        batch = builder.create<AddTileToBatchOp>(
            loc, batchType, batch, nopA, nopB,
            builder.getI32IntegerAttr(i),           // slot
            builder.getI32IntegerAttr(-1),          // matrix_id
            builder.getI64IntegerAttr(0),           // m_offset
            builder.getI64IntegerAttr(0),           // n_offset
            builder.getI64IntegerAttr(0),           // k_offset
            builder.getI64IntegerAttr(0),           // actual_m
            builder.getI64IntegerAttr(0),           // actual_n
            builder.getI64IntegerAttr(0));          // actual_k
      } else {
        // Real tile
        batch = builder.create<AddTileToBatchOp>(
            loc, batchType, batch, tile.tileA, tile.tileB,
            builder.getI32IntegerAttr(i),
            builder.getI32IntegerAttr(tile.matrixId),
            builder.getI64IntegerAttr(tile.mOffset),
            builder.getI64IntegerAttr(tile.nOffset),
            builder.getI64IntegerAttr(tile.kOffset),
            builder.getI64IntegerAttr(tile.actualM),
            builder.getI64IntegerAttr(tile.actualN),
            builder.getI64IntegerAttr(tile.actualK));
      }
    }

    // Execute batch
    auto resultType = BatchResultType::get(builder.getContext());
    Value batchResult = builder.create<ExecuteBatchOp>(
        loc, resultType, batch, builder.getStringAttr("matmul"));

    // Extract results and accumulate
    Value updatedAccum = accumulator;
    for (size_t i = 0; i < tiles.size(); ++i) {
      const TileCandidate &tile = tiles[i];

      if (tile.matrixId == -1)
        continue; // Skip NOP

      // Extract result for this tile
      auto tileResultType =
          RankedTensorType::get({tile.actualM, tile.actualN}, builder.getF32Type());
      Value tileResult = builder.create<ExtractResultOp>(
          loc, tileResultType, batchResult, builder.getI32IntegerAttr(i));

      // Read old value from accumulator
      Value mOffset = builder.create<arith::ConstantIndexOp>(loc, tile.mOffset);
      Value nOffset = builder.create<arith::ConstantIndexOp>(loc, tile.nOffset);

      SmallVector<OpFoldResult> offsets = {mOffset, nOffset};
      SmallVector<OpFoldResult> sizes = {builder.getIndexAttr(tile.actualM),
                                         builder.getIndexAttr(tile.actualN)};
      SmallVector<OpFoldResult> strides = {builder.getIndexAttr(1),
                                           builder.getIndexAttr(1)};

      Value oldValue = builder.create<tensor::ExtractSliceOp>(
          loc, updatedAccum, offsets, sizes, strides);

      // Accumulate: newValue = oldValue + tileResult
      Value emptyTensor = builder.create<tensor::EmptyOp>(
          loc, tileResultType, ValueRange{});
      Value newValue =
          builder.create<linalg::AddOp>(loc, ValueRange{oldValue, tileResult},
                                        ValueRange{emptyTensor})
              .getResult(0);

      // Write back to accumulator
      updatedAccum = builder.create<tensor::InsertSliceOp>(
          loc, newValue, updatedAccum, offsets, sizes, strides);
    }

    return updatedAccum;
  }
};

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createNPUTileSchedulerPass() {
  return std::make_unique<NPUTileSchedulerPass>();
}

} // namespace npu
} // namespace mlir
