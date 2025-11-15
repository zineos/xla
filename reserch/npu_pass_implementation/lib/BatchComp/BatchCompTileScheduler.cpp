//===- BatchCompTileScheduler.cpp - Batch Comp Tile Scheduler Pass -------===//
//
// Implements the main optimization pass for batch composition
//
//===----------------------------------------------------------------------===//

#include "BatchComp/BatchCompPasses.h"
#include "BatchComp/BatchCompOps.h"
#include "BatchComp/BatchCompDialect.h"

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <vector>
#include <string>

#define DEBUG_TYPE "batch-comp-tile-scheduler"

namespace mlir {
namespace batch_comp {

#define GEN_PASS_DEF_BATCHCOMPTILESCHEDULERPASS
#include "BatchComp/BatchCompPasses.h.inc"

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

/// Information about a single matmul operation
struct MatmulInfo {
  tosa::MatMulOp op;
  Value inputA, inputB, result;
  int64_t M, K, N;
  int matrixId;
};

/// Tile information for optimization
struct TileInfo {
  int tileId;
  int matrixId;
  int64_t mOffset, nOffset, kIteration;
  int64_t actualM, actualN, actualK;

  // For Python interface
  std::array<int64_t, 4> aSlice;  // [m_start, m_end, k_start, k_end]
  std::array<int64_t, 4> bSlice;  // [k_start, k_end, n_start, n_end]
  std::array<int64_t, 4> cSlice;  // [m_start, m_end, n_start, n_end]
  bool accumulate;
};

/// Batch scheduling result from Python optimizer
struct BatchScheduleResult {
  std::vector<std::vector<TileInfo>> batches;
  int numBatches;
  int totalNops;
  float utilization;
  std::string algorithm;
};

//===----------------------------------------------------------------------===//
// Python Integration
//===----------------------------------------------------------------------===//

/// Python optimizer interface
///
/// This class provides C++ interface to Python batch optimization algorithms.
/// It uses the Python C API or pybind11 to call Python code.
class PythonOptimizer {
public:
  PythonOptimizer(const std::string &hardwareType,
                  const std::string &configPath = "")
      : hardwareType_(hardwareType), configPath_(configPath) {
    // TODO: Initialize Python interpreter and import modules
    // This would use pybind11 or Python C API
  }

  /// Optimize tile-to-batch assignment
  ///
  /// Calls Python unified optimizer with the given tiles and algorithm.
  ///
  /// \param tiles List of tiles to optimize
  /// \param algorithm Algorithm to use ("auto", "greedy", "improved_greedy", "cpsat")
  /// \return Batch schedule result
  BatchScheduleResult optimize(const std::vector<TileInfo> &tiles,
                                const std::string &algorithm) {
    // TODO: Call Python optimizer
    // For now, return a simple greedy result
    return simpleGreedySchedule(tiles);
  }

private:
  std::string hardwareType_;
  std::string configPath_;

  /// Simple greedy scheduling (placeholder)
  BatchScheduleResult simpleGreedySchedule(const std::vector<TileInfo> &tiles) {
    BatchScheduleResult result;

    // Simple batching: 16 tiles per batch
    const int batchSize = 16;
    std::vector<TileInfo> currentBatch;

    for (const auto &tile : tiles) {
      currentBatch.push_back(tile);

      if (currentBatch.size() == batchSize) {
        result.batches.push_back(currentBatch);
        currentBatch.clear();
      }
    }

    // Handle remaining tiles
    if (!currentBatch.empty()) {
      result.batches.push_back(currentBatch);
    }

    result.numBatches = result.batches.size();
    result.totalNops = 0;
    result.utilization = static_cast<float>(tiles.size()) /
                         (result.numBatches * batchSize);
    result.algorithm = "simple_greedy";

    return result;
  }
};

//===----------------------------------------------------------------------===//
// Tile Generation
//===----------------------------------------------------------------------===//

/// Generate tiles for a matmul operation
std::vector<TileInfo> generateTilesForMatmul(const MatmulInfo &matmul,
                                               ArrayRef<int64_t> tileSize) {
  std::vector<TileInfo> tiles;

  int64_t tileM = tileSize[0];
  int64_t tileN = tileSize[1];
  int64_t tileK = tileSize[2];

  int tileId = 0;

  // Iterate over M dimension
  for (int64_t m = 0; m < matmul.M; m += tileM) {
    int64_t actualM = std::min(tileM, matmul.M - m);

    // Iterate over N dimension
    for (int64_t n = 0; n < matmul.N; n += tileN) {
      int64_t actualN = std::min(tileN, matmul.N - n);

      // Iterate over K dimension
      for (int64_t k = 0; k < matmul.K; k += tileK) {
        int64_t actualK = std::min(tileK, matmul.K - k);

        TileInfo tile;
        tile.tileId = tileId++;
        tile.matrixId = matmul.matrixId;
        tile.mOffset = m;
        tile.nOffset = n;
        tile.kIteration = k / tileK;
        tile.actualM = actualM;
        tile.actualN = actualN;
        tile.actualK = actualK;

        // Set slice information
        tile.aSlice = {m, m + actualM, k, k + actualK};
        tile.bSlice = {k, k + actualK, n, n + actualN};
        tile.cSlice = {m, m + actualM, n, n + actualN};

        // Accumulate if not the first K iteration
        tile.accumulate = (k > 0);

        tiles.push_back(tile);
      }
    }
  }

  return tiles;
}

//===----------------------------------------------------------------------===//
// IR Generation
//===----------------------------------------------------------------------===//

/// Generate high-level BatchComp IR from schedule result
void generateBatchCompIR(OpBuilder &builder, Location loc,
                         const MatmulInfo &matmul,
                         const BatchScheduleResult &schedule,
                         const std::string &hardware,
                         ArrayRef<int64_t> tileSize,
                         int batchSize,
                         const std::string &algorithm) {
  builder.setInsertionPoint(matmul.op);

  LLVM_DEBUG(llvm::dbgs() << "Generating BatchComp IR for matmul "
                          << matmul.matrixId << "\n";
             llvm::dbgs() << "  Batches: " << schedule.numBatches << "\n";
             llvm::dbgs() << "  Algorithm: " << schedule.algorithm << "\n";
             llvm::dbgs() << "  Utilization: " << schedule.utilization << "\n";);

  // Step 1: Create TileSet type
  auto tilesetType = builder.getType<batch_comp::TileSetType>();

  // Step 2: Generate generate_tiles operation
  // %tileset = batch_comp.generate_tiles %A, %B {
  //   tile_size = [16, 16, 16],
  //   matrix_id = 0
  // }
  auto tileSizeAttr = builder.getI64ArrayAttr(tileSize);
  auto matrixIdAttr = builder.getI32IntegerAttr(matmul.matrixId);

  auto generateTilesOp = builder.create<batch_comp::GenerateTilesOp>(
      loc,
      tilesetType,                    // result type
      matmul.inputA,                  // lhs
      matmul.inputB,                  // rhs
      tileSizeAttr,                   // tile_size
      matrixIdAttr                    // matrix_id
  );

  LLVM_DEBUG(llvm::dbgs() << "  Created generate_tiles op\n");

  // Step 3: Create BatchSchedule type
  auto scheduleType = builder.getType<batch_comp::BatchScheduleType>();

  // Step 4: Generate schedule_batches operation
  // %schedule = batch_comp.schedule_batches %tileset {
  //   hardware = "npu",
  //   batch_size = 16,
  //   algorithm = "cpsat"
  // }
  auto hardwareAttr = builder.getStringAttr(hardware);
  auto batchSizeAttr = builder.getI32IntegerAttr(batchSize);
  auto algorithmAttr = builder.getStringAttr(algorithm);

  auto scheduleBatchesOp = builder.create<batch_comp::ScheduleBatchesOp>(
      loc,
      scheduleType,                   // result type
      generateTilesOp.getResult(),    // tiles (TileSet)
      hardwareAttr,                   // hardware
      batchSizeAttr,                  // batch_size
      algorithmAttr                   // algorithm
  );

  LLVM_DEBUG(llvm::dbgs() << "  Created schedule_batches op\n");

  // Step 5: Generate execute_schedule operation
  // %C = batch_comp.execute_schedule %schedule {
  //   hardware = "npu"
  // }
  auto executeScheduleOp = builder.create<batch_comp::ExecuteScheduleOp>(
      loc,
      matmul.result.getType(),         // result type (same as original matmul)
      scheduleBatchesOp.getResult(),   // schedule (BatchSchedule)
      hardwareAttr                     // hardware
  );

  LLVM_DEBUG(llvm::dbgs() << "  Created execute_schedule op\n");

  // Step 6: Replace the original matmul with the execute_schedule result
  matmul.op.getResult().replaceAllUsesWith(executeScheduleOp.getResult());

  // Step 7: Erase the original matmul operation
  matmul.op.erase();

  LLVM_DEBUG(llvm::dbgs() << "  Replaced original matmul with BatchComp IR\n");
}

//===----------------------------------------------------------------------===//
// BatchCompTileSchedulerPass
//===----------------------------------------------------------------------===//

class BatchCompTileSchedulerPass
    : public impl::BatchCompTileSchedulerPassBase<BatchCompTileSchedulerPass> {
public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(context);

    LLVM_DEBUG(llvm::dbgs() << "Running BatchComp Tile Scheduler Pass\n");

    // Get tile size from options (default: [16, 16, 16])
    SmallVector<int64_t, 3> tileSizeVec;
    if (tileSize.empty()) {
      tileSizeVec = {16, 16, 16};
    } else {
      tileSizeVec.assign(tileSize.begin(), tileSize.end());
    }

    // Step 1: Collect all matmul operations
    SmallVector<MatmulInfo> matmuls;
    int matrixId = 0;

    funcOp.walk([&](tosa::MatMulOp matmulOp) {
      auto aType = matmulOp.getA().getType().dyn_cast<RankedTensorType>();
      auto bType = matmulOp.getB().getType().dyn_cast<RankedTensorType>();
      auto cType = matmulOp.getResult().getType().dyn_cast<RankedTensorType>();

      if (!aType || !bType || !cType)
        return;

      ArrayRef<int64_t> aShape = aType.getShape();
      ArrayRef<int64_t> bShape = bType.getShape();

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

      LLVM_DEBUG(llvm::dbgs() << "Found matmul " << info.matrixId << ": "
                              << M << "x" << K << " × " << K << "x" << N << "\n");
    });

    if (matmuls.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "No matmul operations found\n");
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Found " << matmuls.size() << " matmul ops\n");

    // Step 2: Generate tiles for all matmuls
    std::vector<TileInfo> allTiles;
    for (const auto &matmul : matmuls) {
      auto tiles = generateTilesForMatmul(matmul, tileSizeVec);
      allTiles.insert(allTiles.end(), tiles.begin(), tiles.end());

      LLVM_DEBUG(llvm::dbgs() << "Generated " << tiles.size() << " tiles for matmul "
                              << matmul.matrixId << "\n");
    }

    LLVM_DEBUG(llvm::dbgs() << "Total tiles: " << allTiles.size() << "\n");

    // Step 3: Call Python optimizer
    PythonOptimizer optimizer(hardware, configPath);
    BatchScheduleResult schedule = optimizer.optimize(allTiles, algorithm);

    LLVM_DEBUG(llvm::dbgs() << "Optimization complete:\n";
               llvm::dbgs() << "  Algorithm: " << schedule.algorithm << "\n";
               llvm::dbgs() << "  Batches: " << schedule.numBatches << "\n";
               llvm::dbgs() << "  Utilization: " << schedule.utilization << "\n";);

    // Determine batch size based on hardware type
    int batchSize = 16;  // default NPU
    if (hardware == "gpu") {
      batchSize = 32;
    } else if (hardware == "tpu") {
      batchSize = 128;
    }

    // Step 4: Generate BatchComp IR for each matmul
    for (const auto &matmul : matmuls) {
      generateBatchCompIR(builder, matmul.op.getLoc(), matmul, schedule,
                          hardware, tileSizeVec, batchSize, algorithm);
    }

    LLVM_DEBUG(llvm::dbgs() << "BatchComp Tile Scheduler Pass complete\n");
  }
};

//===----------------------------------------------------------------------===//
// Pass Registration
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> createBatchCompTileSchedulerPass() {
  return std::make_unique<BatchCompTileSchedulerPass>();
}

} // namespace batch_comp
} // namespace mlir
