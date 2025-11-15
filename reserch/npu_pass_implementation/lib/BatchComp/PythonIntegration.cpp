//===- PythonIntegration.cpp - Python-C++ Integration Layer --------------===//
//
// Provides interface for calling Python batch composition optimizers from C++
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"
#include "llvm/ADT/SmallVector.h"

#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

#define DEBUG_TYPE "python-integration"

// Conditional Python integration based on availability
#ifdef ENABLE_PYBIND11
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;
#define HAS_PYTHON_SUPPORT 1
#else
#define HAS_PYTHON_SUPPORT 0
#endif

namespace mlir {
namespace batch_comp {

//===----------------------------------------------------------------------===//
// Data Structures (matching Python interface)
//===----------------------------------------------------------------------===//

/// Tile information for Python optimizer
struct TileInfo {
  int tileId;
  int matrixId;
  int64_t mOffset, nOffset, kIteration;
  int64_t actualM, actualN, actualK;
  std::array<int64_t, 4> aSlice;
  std::array<int64_t, 4> bSlice;
  std::array<int64_t, 4> cSlice;
  bool accumulate;

  /// Convert to Python dictionary
  #if HAS_PYTHON_SUPPORT
  py::dict toPython() const {
    py::dict tile;
    tile["tile_id"] = tileId;
    tile["matrix_id"] = matrixId;
    tile["m_offset"] = mOffset;
    tile["n_offset"] = nOffset;
    tile["k_iteration"] = kIteration;
    tile["actual_m"] = actualM;
    tile["actual_n"] = actualN;
    tile["actual_k"] = actualK;

    // Convert arrays to Python lists
    tile["a_slice"] = py::cast(aSlice);
    tile["b_slice"] = py::cast(bSlice);
    tile["c_slice"] = py::cast(cSlice);
    tile["accumulate"] = accumulate;

    return tile;
  }
  #endif
};

/// Batch schedule result from Python
struct BatchScheduleResult {
  std::vector<std::vector<int>> batches;  // List of tile IDs per batch
  int numBatches;
  int totalNops;
  float utilization;
  std::string algorithm;
  double executionTime;

  #if HAS_PYTHON_SUPPORT
  /// Create from Python dictionary
  static BatchScheduleResult fromPython(const py::dict &pyResult) {
    BatchScheduleResult result;

    // Extract batches (list of lists of ints)
    py::list pyBatches = pyResult["batches"];
    for (const auto &pyBatch : pyBatches) {
      std::vector<int> batch = pyBatch.cast<std::vector<int>>();
      result.batches.push_back(batch);
    }

    result.numBatches = pyResult["num_batches"].cast<int>();
    result.totalNops = pyResult["total_nops"].cast<int>();
    result.utilization = pyResult["utilization"].cast<float>();
    result.algorithm = pyResult["algorithm"].cast<std::string>();
    result.executionTime = pyResult["execution_time"].cast<double>();

    return result;
  }
  #endif
};

//===----------------------------------------------------------------------===//
// PythonOptimizer - Main Integration Class
//===----------------------------------------------------------------------===//

class PythonOptimizer {
public:
  /// Constructor
  ///
  /// \param hardwareType Hardware platform ("npu", "gpu", "tpu")
  /// \param configPath Optional path to hardware config YAML file
  PythonOptimizer(const std::string &hardwareType,
                  const std::string &configPath = "")
      : hardwareType_(hardwareType), configPath_(configPath),
        pythonAvailable_(false) {

    #if HAS_PYTHON_SUPPORT
    try {
      initializePython();
      pythonAvailable_ = true;
    } catch (const std::exception &e) {
      LLVM_DEBUG(llvm::dbgs() << "Python initialization failed: " << e.what()
                              << "\n");
      pythonAvailable_ = false;
    }
    #else
    LLVM_DEBUG(llvm::dbgs() << "Python support not compiled in\n");
    #endif
  }

  /// Check if Python optimizer is available
  bool isAvailable() const { return pythonAvailable_; }

  /// Optimize tile-to-batch assignment
  ///
  /// \param tiles List of tiles to optimize
  /// \param algorithm Algorithm to use ("auto", "greedy", "improved_greedy", "cpsat")
  /// \return Batch schedule result
  BatchScheduleResult optimize(const std::vector<TileInfo> &tiles,
                                const std::string &algorithm = "auto") {
    #if HAS_PYTHON_SUPPORT
    if (!pythonAvailable_) {
      return fallbackSchedule(tiles);
    }

    try {
      return optimizeWithPython(tiles, algorithm);
    } catch (const std::exception &e) {
      LLVM_DEBUG(llvm::dbgs() << "Python optimization failed: " << e.what()
                              << "\nFalling back to simple scheduler\n");
      return fallbackSchedule(tiles);
    }
    #else
    return fallbackSchedule(tiles);
    #endif
  }

private:
  std::string hardwareType_;
  std::string configPath_;
  bool pythonAvailable_;

  #if HAS_PYTHON_SUPPORT
  py::object pyInterface_;

  /// Initialize Python interpreter and import modules
  void initializePython() {
    // Import the C++ interface module
    py::module interface = py::module::import("python.interface.cpp_interface");

    // Create hardware config
    py::dict hwConfig;
    hwConfig["hardware_type"] = hardwareType_;
    if (!configPath_.empty()) {
      hwConfig["config_path"] = configPath_;
    }

    // Initialize the interface
    interface.attr("initialize_interface")(hwConfig);

    pyInterface_ = interface;

    LLVM_DEBUG(llvm::dbgs() << "Python optimizer initialized successfully\n");
  }

  /// Optimize using Python
  BatchScheduleResult optimizeWithPython(const std::vector<TileInfo> &tiles,
                                          const std::string &algorithm) {
    // Convert tiles to Python format
    py::list pyTiles;
    for (const auto &tile : tiles) {
      pyTiles.append(tile.toPython());
    }

    // Call Python optimizer
    py::object optimizeFunc = pyInterface_.attr("optimize_tiles");
    py::dict pyResult = optimizeFunc(pyTiles, algorithm);

    // Check for errors
    if (pyResult.contains("error")) {
      std::string error = pyResult["error"].cast<std::string>();
      throw std::runtime_error("Python optimization error: " + error);
    }

    // Convert result back to C++
    return BatchScheduleResult::fromPython(pyResult);
  }
  #endif

  /// Fallback simple greedy scheduler (when Python is unavailable)
  ///
  /// This is a simple C++-only implementation for cases where Python
  /// is not available or fails.
  BatchScheduleResult fallbackSchedule(const std::vector<TileInfo> &tiles) {
    BatchScheduleResult result;

    const int batchSize = 16;  // Default NPU batch size
    std::vector<int> currentBatch;

    for (const auto &tile : tiles) {
      currentBatch.push_back(tile.tileId);

      if (currentBatch.size() == batchSize) {
        result.batches.push_back(currentBatch);
        currentBatch.clear();
      }
    }

    // Add remaining tiles
    if (!currentBatch.empty()) {
      result.batches.push_back(currentBatch);
    }

    result.numBatches = result.batches.size();
    result.totalNops = 0;

    if (result.numBatches > 0) {
      int totalTiles = tiles.size();
      int totalSlots = result.numBatches * batchSize;
      result.utilization = static_cast<float>(totalTiles) / totalSlots;
      result.totalNops = totalSlots - totalTiles;
    } else {
      result.utilization = 0.0f;
    }

    result.algorithm = "fallback_greedy";
    result.executionTime = 0.0;

    LLVM_DEBUG(llvm::dbgs() << "Using fallback scheduler: "
                            << result.numBatches << " batches, "
                            << result.utilization * 100 << "% utilization\n");

    return result;
  }
};

//===----------------------------------------------------------------------===//
// Alternative: JSON-based Integration (simpler, no pybind11 required)
//===----------------------------------------------------------------------===//

#include <fstream>
#include <sstream>
#include <cstdlib>

/// JSON-based Python optimizer (alternative to pybind11)
///
/// This class uses JSON files to communicate with Python, avoiding the
/// need for pybind11. It's simpler but has more overhead.
class JSONPythonOptimizer {
public:
  JSONPythonOptimizer(const std::string &hardwareType,
                      const std::string &configPath = "")
      : hardwareType_(hardwareType), configPath_(configPath) {}

  BatchScheduleResult optimize(const std::vector<TileInfo> &tiles,
                                const std::string &algorithm = "auto") {
    // 1. Write tiles to JSON file
    std::string inputFile = "/tmp/tiles_input.json";
    writeTilesToJSON(tiles, algorithm, inputFile);

    // 2. Call Python script
    std::string pythonScript =
        "python/interface/cpp_interface.py";  // TODO: full path
    std::string outputFile = "/tmp/tiles_output.json";

    std::string command =
        "python3 -c \"import json; "
        "from python.interface.cpp_interface import optimize_tiles_from_json; "
        "with open('" + inputFile + "') as f: data = f.read(); "
        "result = optimize_tiles_from_json(data); "
        "with open('" + outputFile + "', 'w') as f: f.write(result)\"";

    int ret = std::system(command.c_str());

    if (ret != 0) {
      throw std::runtime_error("Python script execution failed");
    }

    // 3. Read result from JSON
    return readResultFromJSON(outputFile);
  }

private:
  std::string hardwareType_;
  std::string configPath_;

  void writeTilesToJSON(const std::vector<TileInfo> &tiles,
                        const std::string &algorithm,
                        const std::string &filename) {
    std::ofstream out(filename);

    out << "{\n";
    out << "  \"algorithm\": \"" << algorithm << "\",\n";
    out << "  \"hardware_config\": {\"hardware_type\": \""
        << hardwareType_ << "\"},\n";
    out << "  \"tiles\": [\n";

    for (size_t i = 0; i < tiles.size(); ++i) {
      const auto &tile = tiles[i];
      out << "    {\n";
      out << "      \"tile_id\": " << tile.tileId << ",\n";
      out << "      \"matrix_id\": " << tile.matrixId << ",\n";
      out << "      \"m_offset\": " << tile.mOffset << ",\n";
      out << "      \"n_offset\": " << tile.nOffset << ",\n";
      out << "      \"k_iteration\": " << tile.kIteration << ",\n";
      out << "      \"actual_m\": " << tile.actualM << ",\n";
      out << "      \"actual_n\": " << tile.actualN << ",\n";
      out << "      \"actual_k\": " << tile.actualK << ",\n";
      out << "      \"a_slice\": ["
          << tile.aSlice[0] << "," << tile.aSlice[1] << ","
          << tile.aSlice[2] << "," << tile.aSlice[3] << "],\n";
      out << "      \"b_slice\": ["
          << tile.bSlice[0] << "," << tile.bSlice[1] << ","
          << tile.bSlice[2] << "," << tile.bSlice[3] << "],\n";
      out << "      \"c_slice\": ["
          << tile.cSlice[0] << "," << tile.cSlice[1] << ","
          << tile.cSlice[2] << "," << tile.cSlice[3] << "],\n";
      out << "      \"accumulate\": "
          << (tile.accumulate ? "true" : "false") << "\n";
      out << "    }";
      if (i < tiles.size() - 1) out << ",";
      out << "\n";
    }

    out << "  ]\n";
    out << "}\n";
  }

  BatchScheduleResult readResultFromJSON(const std::string &filename) {
    // TODO: Implement proper JSON parsing
    // For now, return a simple result
    BatchScheduleResult result;
    result.numBatches = 0;
    result.totalNops = 0;
    result.utilization = 0.0f;
    result.algorithm = "json_based";
    result.executionTime = 0.0;
    return result;
  }
};

} // namespace batch_comp
} // namespace mlir
