# BatchComp Framework 架构概览

**系统架构设计文档**

---

## 📋 文档信息

- **版本**: v1.0
- **最后更新**: 2025-11-15
- **状态**: 核心完成 (85%)
- **作者**: NPU Compiler Team

---

## 🎯 架构目标

BatchComp框架旨在提供一个**统一的批组合优化抽象**，支持多种tile-based加速器：

1. **统一抽象** - NPU/GPU/TPU共用同一套优化框架
2. **算法可选** - 从简单贪心到全局CP-SAT优化
3. **语言分离** - Python负责优化，C++负责编译
4. **两级IR** - 高层简洁表达，低层精确控制

---

## 🏗️ 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Application                          │
│                   (TOSA/Linalg MLIR IR)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 BatchComp MLIR Pass Pipeline                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  BatchCompTileSchedulerPass                               │  │
│  │  - 收集matmul操作                                          │  │
│  │  - 生成tiles                                               │  │
│  │  - 调用Python优化器 ◄────┐                                 │  │
│  │  - 生成高层BatchComp IR   │                                │  │
│  └───────────────────────────┼───────────────────────────────┘  │
│                              │                                   │
│  ┌───────────────────────────▼───────────────────────────────┐  │
│  │  Python Optimizer                                         │  │
│  │  - Greedy / ImprovedGreedy / CP-SAT                       │  │
│  │  - 硬件配置驱动                                            │  │
│  │  - 返回batch调度结果                                       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                              │                                   │
│                              ▼                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  BatchCompLoweringPass                                    │  │
│  │  - 高层IR → 低层IR                                         │  │
│  │  - generate_tiles → slice + pad                           │  │
│  │  - execute_schedule → batch + execute + accumulate        │  │
│  └───────────────────────┬───────────────────────────────────┘  │
│                          │                                       │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  BatchCompToNPUPass / ToGPUPass / ToTPUPass               │  │
│  │  - 转换到硬件特定Dialect                                   │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Hardware-Specific Backend                      │
│                   (NPU / GPU / TPU Dialect)                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🧩 核心组件

### 1. Python优化器层

**位置**: `python/`

**职责**: 实现批组合优化算法

```
python/
├── core/
│   ├── tile_info.py           # Tile数据结构
│   ├── hardware_config.py     # 硬件配置系统
│   └── batch_optimizer.py     # 优化器基类
├── optimizers/
│   ├── greedy.py              # O(n) 简单贪心
│   ├── improved_greedy.py     # O(n log n) K-dimension分组
│   └── cpsat_optimizer.py     # NP 全局优化
└── interface/
    └── cpp_interface.py       # C++调用接口
```

**关键特性**:
- 硬件配置驱动（YAML配置文件）
- 多算法支持（auto-selection）
- K-dimension分组优化
- 跨矩阵批处理

### 2. MLIR Dialect层

**位置**: `include/BatchComp/`

**职责**: 定义BatchComp MLIR操作

**两级IR设计**:

#### 高层IR（用户友好）
```mlir
%tiles = batch_comp.generate_tiles %A, %B {tile_size = [16,16,16]}
%schedule = batch_comp.schedule_batches %tiles {hardware = "npu"}
%C = batch_comp.execute_schedule %schedule
```

#### 低层IR（精确控制）
```mlir
%tile_a = batch_comp.slice %A[0, 0] [16, 16]
%batch = batch_comp.create_batch {batch_size = 16}
%batch1 = batch_comp.add_to_batch %batch, %tile_a, %tile_b {...}
%result = batch_comp.execute_batch %batch1
```

**15个操作**:
- **Tile生成**: `slice`, `pad`, `tile_matmul`, `accumulate`
- **Batch管理**: `create_batch`, `add_to_batch`, `execute_batch`
- **高层抽象**: `generate_tiles`, `schedule_batches`, `execute_schedule`
- **工具**: `nop_tile`

### 3. C++ Pass层

**位置**: `lib/BatchComp/`

**职责**: IR转换和优化

**4个核心Pass**:

1. **BatchCompTileSchedulerPass**
   - 入口Pass，处理TOSA/Linalg matmul
   - 调用Python优化器
   - 生成高层BatchComp IR

2. **BatchCompLoweringPass**
   - 高层IR → 低层IR转换
   - 处理K-dimension累加
   - 处理边缘tile padding

3. **BatchCompToNPUPass**
   - 转换到NPU Dialect
   - 硬件特定优化

4. **BatchCompCanonicalizePass**
   - IR规范化和优化
   - 消除冗余操作

### 4. Python-C++集成层

**位置**: `lib/BatchComp/PythonIntegration.cpp`

**职责**: 双向数据转换和调用

**三种集成方案**:

1. **pybind11** (推荐)
   ```cpp
   py::module interface = py::module::import("python.interface.cpp_interface");
   py::dict result = interface.attr("optimize_tiles")(tiles, algorithm);
   ```

2. **JSON** (备选)
   ```cpp
   writeTilesToJSON(tiles, "input.json");
   system("python3 optimizer.py");
   readResultFromJSON("output.json");
   ```

3. **C++ Fallback** (鲁棒性)
   ```cpp
   // 当Python不可用时，使用简单C++贪心调度
   return fallbackSchedule(tiles);
   ```

---

## 🔄 数据流

### 完整编译流程

```
1. TOSA Matmul
   ↓
2. Collect matmuls (BatchCompTileSchedulerPass)
   ↓
3. Generate TileInfo structures
   ↓
4. Call Python optimizer via pybind11
   ↓
5. Python returns BatchScheduleResult
   ↓
6. Generate high-level BatchComp IR
   %tiles = batch_comp.generate_tiles ...
   %schedule = batch_comp.schedule_batches ...
   %result = batch_comp.execute_schedule ...
   ↓
7. Lower to detailed IR (BatchCompLoweringPass)
   %tile = batch_comp.slice ...
   %batch = batch_comp.create_batch ...
   %batch1 = batch_comp.add_to_batch ...
   %result = batch_comp.execute_batch ...
   ↓
8. Convert to hardware dialect (BatchCompToNPUPass)
   %npu_batch = npu.create_batch ...
   %result = npu.parallel_execute ...
   ↓
9. Backend code generation
```

### Tile信息传递

```cpp
// C++ → Python
struct TileInfo {
    int tile_id, matrix_id;
    int64_t m_offset, n_offset, k_iteration;
    int64_t actual_m, actual_n, actual_k;
    std::array<int64_t, 4> a_slice, b_slice, c_slice;
    bool accumulate;
};

// Python TileInfo dataclass
@dataclass
class TileInfo:
    tile_id: int
    matrix_id: int
    ...同C++
```

### 批调度结果返回

```cpp
// Python → C++
struct BatchScheduleResult {
    std::vector<std::vector<int>> batches;  // 每个batch的tile IDs
    int numBatches;
    int totalNops;
    float utilization;
    std::string algorithm;
    double executionTime;
};
```

---

## 🎨 设计模式

### 1. 策略模式（Optimization Algorithms）

```python
class BatchOptimizer(ABC):
    @abstractmethod
    def optimize(self, tiles, **kwargs) -> BatchScheduleResult:
        pass

class GreedyOptimizer(BatchOptimizer): ...
class ImprovedGreedyOptimizer(BatchOptimizer): ...
class CPSATOptimizer(BatchOptimizer): ...

# Auto-selection
def UnifiedOptimizer(...):
    if num_tiles < 30: return GreedyOptimizer()
    elif num_tiles < 100: return ImprovedGreedyOptimizer()
    else: return CPSATOptimizer()
```

### 2. 工厂模式（Hardware Configuration）

```python
class HardwareConfig:
    @classmethod
    def from_yaml(cls, path: str) -> HardwareConfig: ...

    @classmethod
    def get_default(cls, hw_type: str) -> HardwareConfig:
        if hw_type == "npu": return NPU_DEFAULT_CONFIG
        elif hw_type == "gpu": return GPU_DEFAULT_CONFIG
        elif hw_type == "tpu": return TPU_DEFAULT_CONFIG
```

### 3. 访问者模式（MLIR Pattern Rewriting）

```cpp
class GenerateTilesLowering : public OpRewritePattern<GenerateTilesOp> {
    LogicalResult matchAndRewrite(GenerateTilesOp op, PatternRewriter &rewriter) {
        // 将generate_tiles展开为slice + pad操作
    }
};
```

---

## 🔧 扩展点

### 1. 添加新的优化算法

```python
# python/optimizers/my_algorithm.py
class MyOptimizer(BatchOptimizer):
    def optimize(self, tiles, **kwargs):
        # 实现你的算法
        return BatchScheduleResult(...)
```

### 2. 支持新的硬件平台

```yaml
# configs/my_hardware.yaml
name: "My Accelerator"
hardware_type: "custom"
compute:
  batch_size: 32
  ...
```

```cpp
// lib/BatchComp/BatchCompToMyHW.cpp
class BatchCompToMyHWPass : public PassWrapper<...> {
    // 实现转换
};
```

### 3. 添加新的Dialect操作

```tablegen
// include/BatchComp/BatchCompOps.td
def BatchComp_MyOp : BatchComp_Op<"my_op", [Pure]> {
    let summary = "My custom operation";
    ...
};
```

---

## 📊 性能特征

### 算法复杂度

| 算法 | 时间复杂度 | 空间复杂度 | 最优性 |
|------|-----------|-----------|--------|
| Greedy | O(n) | O(n) | 局部最优 |
| ImprovedGreedy | O(n log n) | O(n) | 较好 |
| CP-SAT | NP (有时限) | O(n²) | 全局最优* |

*在时间限制内找到的最优解

### 适用规模

| 算法 | 适用tile数 | 适用场景 |
|------|-----------|---------|
| Greedy | < 50 | 单矩阵，快速编译 |
| ImprovedGreedy | 50-500 | 多矩阵，生产环境 |
| CP-SAT | 50-1000 | 离线优化，追求极致性能 |

---

## 🔒 设计原则

1. **关注点分离**
   - Python: 复杂优化算法
   - C++: MLIR编译基础设施

2. **可测试性**
   - Python优化器独立测试
   - MLIR Pass独立测试
   - 集成测试分离

3. **可扩展性**
   - 新算法：继承BatchOptimizer
   - 新硬件：添加YAML配置
   - 新操作：扩展TableGen定义

4. **鲁棒性**
   - Python失败 → C++ fallback
   - 配置缺失 → 使用默认值
   - 硬件未知 → 通用优化

---

## 📚 相关文档

- [Dialect完整设计](BATCHCOMP_DIALECT_DESIGN.md)
- [Python-C++集成](PYTHON_CPP_INTEGRATION.md)
- [Phase 1: Python重构](../summaries/PHASE1_SUMMARY.md)
- [Phase 2: Dialect设计](../summaries/PHASE2_SUMMARY.md)
- [Phase 3: Pass实现](../summaries/PHASE3_PROGRESS.md)

---

**最后更新**: 2025-11-15
**架构版本**: v1.0
**实现进度**: 85%
