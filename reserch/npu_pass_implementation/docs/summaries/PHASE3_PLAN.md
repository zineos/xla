# Phase 3 实施计划 - C++ MLIR Pass实现

**日期**: 2025-11-15
**状态**: 🚧 进行中
**预计完成**: Week 3-4

---

## 目标

实现BatchComp Dialect的核心MLIR Pass，包括：
1. BatchCompTileSchedulerPass - 主优化Pass
2. BatchCompLoweringPass - IR lowering
3. Python-C++集成层
4. 端到端测试

---

## 1. 架构概览

```
┌─────────────────────────────────────────────────────────┐
│                   Input: TOSA/Linalg                    │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│          BatchCompTileSchedulerPass (C++)               │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Collect matmul operations                      │  │
│  │ 2. Generate TileInfo for each matmul             │  │
│  │ 3. Call Python optimizer                         │  │
│  │ 4. Generate BatchComp high-level IR              │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│          BatchComp High-level IR                        │
│  - generate_tiles                                       │
│  - schedule_batches                                     │
│  - execute_schedule                                     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│          BatchCompLoweringPass (C++)                    │
│  ┌───────────────────────────────────────────────────┐  │
│  │ 1. Lower generate_tiles → slice + pad ops        │  │
│  │ 2. Lower schedule_batches → create_batch ops     │  │
│  │ 3. Lower execute_schedule → execute_batch ops    │  │
│  └───────────────────────────────────────────────────┘  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│          BatchComp Low-level IR                         │
│  - slice, pad, tile_matmul, accumulate                 │
│  - create_batch, add_to_batch, execute_batch           │
└────────────────────┬────────────────────────────────────┘
                     │
                     ↓
┌─────────────────────────────────────────────────────────┐
│          Hardware Backend Pass                          │
│  - BatchCompToNPUPass                                   │
│  - BatchCompToGPUPass (future)                         │
└─────────────────────────────────────────────────────────┘
```

---

## 2. 文件结构

```
npu_pass_implementation/
├── include/BatchComp/
│   ├── BatchCompDialect.h           # Dialect注册
│   ├── BatchCompOps.h                # 操作声明
│   ├── BatchCompOps.td               # ✅ 已完成
│   ├── BatchCompPasses.h             # Pass声明
│   └── BatchCompPasses.td            # Pass TableGen定义
│
├── lib/BatchComp/
│   ├── Dialect/
│   │   ├── BatchCompDialect.cpp      # Dialect实现
│   │   └── BatchCompOps.cpp          # 操作实现
│   ├── Transforms/
│   │   ├── BatchCompTileScheduler.cpp  # 主Pass
│   │   ├── BatchCompLowering.cpp       # Lowering Pass
│   │   └── TileGenerator.cpp           # Tile生成逻辑
│   └── Python/
│       └── PythonIntegration.cpp       # Python接口
│
├── python/
│   └── interface/
│       └── cpp_interface.py            # C++调用接口
│
└── tests/
    ├── BatchComp/
    │   ├── tile-scheduler.mlir         # 端到端测试
    │   └── lowering.mlir                # Lowering测试
    └── Python/
        └── test_cpp_integration.py      # Python集成测试
```

---

## 3. 实施步骤

### 3.1 Step 1: Dialect基础设施（1-2天）

**目标**: 搭建BatchComp Dialect的C++框架

**任务**:
- [ ] 创建`BatchCompDialect.h/.cpp`
- [ ] 创建`BatchCompOps.h/.cpp`
- [ ] 创建`BatchCompPasses.td`
- [ ] CMakeLists.txt配置
- [ ] 编译验证

**产出**:
```cpp
// BatchCompDialect.h
namespace mlir {
namespace batch_comp {

class BatchCompDialect : public Dialect {
public:
  explicit BatchCompDialect(MLIRContext *context);
  static StringRef getDialectNamespace() { return "batch_comp"; }

  void initialize();
};

} // namespace batch_comp
} // namespace mlir
```

### 3.2 Step 2: BatchCompTileSchedulerPass（3-4天）

**目标**: 实现主优化Pass

**核心逻辑**:
```cpp
void BatchCompTileSchedulerPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  // 1. Collect matmul ops
  SmallVector<MatmulInfo> matmuls = collectMatmuls(funcOp);

  // 2. Generate tiles
  std::vector<TileInfo> tiles = generateTiles(matmuls);

  // 3. Call Python optimizer
  PythonOptimizer optimizer(hardwareConfig);
  BatchScheduleResult schedule = optimizer.optimize(tiles, "auto");

  // 4. Generate high-level BatchComp IR
  generateHighLevelIR(schedule);
}
```

**任务**:
- [ ] Matmul收集逻辑
- [ ] Tile生成逻辑
- [ ] Python调用接口
- [ ] IR生成逻辑
- [ ] 单元测试

### 3.3 Step 3: Python-C++集成（2-3天）

**目标**: 实现Python优化器的C++调用接口

**接口设计**:
```cpp
// PythonIntegration.h
class PythonOptimizer {
public:
  PythonOptimizer(const HardwareConfig &config);

  BatchScheduleResult optimize(
    const std::vector<TileInfo> &tiles,
    const std::string &algorithm
  );

private:
  pybind11::object py_optimizer_;
};
```

**实现方式**:
- 选项1: pybind11 (推荐)
- 选项2: Python C API
- 选项3: 外部进程调用

**任务**:
- [ ] 选择集成方式
- [ ] 实现C++ wrapper
- [ ] 数据结构转换（C++ ↔ Python）
- [ ] 错误处理
- [ ] 性能测试

### 3.4 Step 4: BatchCompLoweringPass（2-3天）

**目标**: 将高层IR展开为底层IR

**Lowering规则**:

| 高层操作 | 底层操作 |
|----------|----------|
| `generate_tiles` | 多个`slice` + `pad` |
| `schedule_batches` | `create_batch` + `add_to_batch` |
| `execute_schedule` | 多个`execute_batch` + `accumulate` |

**实现**:
```cpp
void lowerGenerateTilesOp(GenerateTilesOp op) {
  // For each tile position:
  for (int m = 0; m < M; m += 16) {
    for (int n = 0; n < N; n += 16) {
      for (int k = 0; k < K; k += 16) {
        // Create slice ops
        auto tile_a = builder.create<SliceOp>(...);
        auto tile_b = builder.create<SliceOp>(...);

        // Pad if needed
        if (needsPadding) {
          tile_a = builder.create<PadOp>(tile_a, ...);
        }
      }
    }
  }
}
```

**任务**:
- [ ] generate_tiles lowering
- [ ] schedule_batches lowering
- [ ] execute_schedule lowering
- [ ] 验证测试

### 3.5 Step 5: 测试和验证（2-3天）

**目标**: 确保端到端正确性

**测试层级**:

1. **单元测试**
   ```mlir
   // Test: slice operation
   %tile = batch_comp.slice %A[0, 0] [16, 16]
   ```

2. **集成测试**
   ```mlir
   // Test: complete matmul transformation
   func.func @matmul(%A, %B) {
     %C = tosa.matmul %A, %B
     return %C
   }
   ```

3. **端到端测试**
   - 小矩阵 (16×16×16)
   - 中矩阵 (100×200×300)
   - 大矩阵 (1024×2048×512)
   - 多矩阵场景

**任务**:
- [ ] 编写MLIR测试用例
- [ ] 编写Python集成测试
- [ ] 性能基准测试
- [ ] 回归测试

---

## 4. Python-C++集成方案

### 方案1: pybind11（推荐）

**优点**:
- ✅ 类型安全
- ✅ 自动类型转换
- ✅ 异常处理
- ✅ 现代C++支持

**示例**:
```cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

class PythonOptimizer {
  py::object optimizer_;

public:
  PythonOptimizer(const std::string &config_path) {
    py::module batch_comp = py::module::import("python.core.batch_optimizer");
    py::object UnifiedOptimizer = batch_comp.attr("UnifiedOptimizer");

    py::module hw_config = py::module::import("python.core.hardware_config");
    py::object config = hw_config.attr("HardwareConfig").attr("from_yaml")(config_path);

    optimizer_ = UnifiedOptimizer(config);
  }

  BatchScheduleResult optimize(const std::vector<TileInfo> &tiles,
                                const std::string &algorithm) {
    // Convert C++ tiles to Python list
    py::list py_tiles;
    for (const auto &tile : tiles) {
      py_tiles.append(toPython(tile));
    }

    // Call Python optimizer
    py::object result = optimizer_.attr("optimize")(py_tiles, algorithm);

    // Convert result back to C++
    return fromPython(result);
  }
};
```

### 方案2: 外部进程（备选）

通过JSON文件交换数据：

```cpp
BatchScheduleResult optimize(const std::vector<TileInfo> &tiles) {
  // 1. Write tiles to JSON
  writeTilesToJSON("tiles.json", tiles);

  // 2. Call Python script
  system("python3 optimize.py --input tiles.json --output schedule.json");

  // 3. Read result from JSON
  return readScheduleFromJSON("schedule.json");
}
```

**优点**: 简单、隔离性好
**缺点**: 性能较差（I/O开销）

---

## 5. 关键数据结构转换

### TileInfo (C++ ↔ Python)

```cpp
// C++ → Python
py::object toPython(const TileInfo &tile) {
  py::module tile_info = py::module::import("python.core.tile_info");
  return tile_info.attr("TileInfo")(
    py::arg("tile_id") = tile.tile_id,
    py::arg("matrix_id") = tile.matrix_id,
    py::arg("m_offset") = tile.m_offset,
    // ... more fields
  );
}

// Python → C++
TileInfo fromPython(py::object py_tile) {
  TileInfo tile;
  tile.tile_id = py_tile.attr("tile_id").cast<int>();
  tile.matrix_id = py_tile.attr("matrix_id").cast<int>();
  // ... more fields
  return tile;
}
```

---

## 6. 性能目标

| 指标 | 目标 |
|------|------|
| **小矩阵 (16×16×16)** | < 0.1s |
| **中矩阵 (100×200×300)** | < 1s (greedy), < 10s (cpsat) |
| **大矩阵 (1024×2048×512)** | < 5s (greedy) |
| **Python调用开销** | < 10ms per call |
| **内存使用** | < 100MB (1000 tiles) |

---

## 7. 验收标准

- [ ] 所有单元测试通过
- [ ] 端到端测试通过（小/中/大矩阵）
- [ ] Python集成测试通过
- [ ] 性能满足目标
- [ ] 代码覆盖率 > 80%
- [ ] 文档完整

---

## 8. 风险和缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Python-C++集成复杂 | 高 | 使用pybind11，提前验证 |
| 性能不达标 | 中 | 优化热点路径，缓存结果 |
| 内存泄漏 | 中 | RAII，智能指针，测试 |
| IR生成错误 | 高 | 充分测试，验证工具 |

---

## 9. 时间线

| Week | 任务 | 状态 |
|------|------|------|
| Week 3 Day 1-2 | Dialect基础设施 | 🚧 |
| Week 3 Day 3-5 | BatchCompTileSchedulerPass | ⏳ |
| Week 4 Day 1-3 | Python-C++集成 | ⏳ |
| Week 4 Day 4-5 | BatchCompLoweringPass | ⏳ |
| Week 5 Day 1-3 | 测试和验证 | ⏳ |
| Week 5 Day 4-5 | 文档和优化 | ⏳ |

---

## 10. 参考资料

- MLIR Pass Infrastructure: https://mlir.llvm.org/docs/PassManagement/
- pybind11 Documentation: https://pybind11.readthedocs.io/
- Phase 1总结: [PHASE1_SUMMARY.md](PHASE1_SUMMARY.md)
- Phase 2总结: [PHASE2_SUMMARY.md](PHASE2_SUMMARY.md)
- Dialect设计: [BATCHCOMP_DIALECT_DESIGN.md](BATCHCOMP_DIALECT_DESIGN.md)

---

**创建时间**: 2025-11-15
**负责人**: NPU Compiler Team
**状态**: 🚧 Phase 3进行中
