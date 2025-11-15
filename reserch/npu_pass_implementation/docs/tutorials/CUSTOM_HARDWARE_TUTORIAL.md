# 自定义硬件支持教程

**为BatchComp框架添加新硬件支持的完整指南**

本教程将指导您如何将BatchComp框架扩展到支持自定义硬件加速器，涵盖从硬件建模到编译器集成的完整流程。

---

## 📋 目录

1. [概述](#1-概述)
2. [硬件特性分析](#2-硬件特性分析)
3. [创建硬件配置文件](#3-创建硬件配置文件)
4. [Python优化器集成](#4-python优化器集成)
5. [MLIR Dialect扩展](#5-mlir-dialect扩展)
6. [后端代码生成](#6-后端代码生成)
7. [测试和验证](#7-测试和验证)
8. [性能调优](#8-性能调优)
9. [完整示例](#9-完整示例)

---

## 1. 概述

### 1.1 支持新硬件的步骤

为BatchComp添加新硬件支持需要以下步骤：

```
1. 硬件特性分析
   ↓
2. 创建硬件配置文件 (YAML)
   ↓
3. Python优化器适配
   ↓
4. MLIR后端代码生成
   ↓
5. 测试和验证
```

### 1.2 案例研究

本教程以一个虚构的硬件为例：**"SynapseAI" 神经网络加速器**

**硬件规格:**
- Tensor Core单元: 8个
- Batch大小: 32个tiles
- Tile维度: 32×32×32 (float16)
- 内存层次: L1 Cache (256KB), L2 Cache (2MB)
- 特殊约束: K维度必须是32的倍数

---

## 2. 硬件特性分析

### 2.1 关键硬件参数

创建硬件配置前，需要明确以下参数：

| 参数类别 | 参数名 | 说明 | SynapseAI示例 |
|---------|--------|------|---------------|
| **计算能力** | max_batch_size | 单次可并行的tile数 | 32 |
| | num_cores | 计算核心数 | 8 |
| | tile_size | 单个tile维度 [M, K, N] | [32, 32, 32] |
| | ops_per_tile | 每个tile的操作数 | 32×32×32×2 = 65536 FLOPs |
| **内存** | l1_cache_kb | L1缓存大小 (KB) | 256 |
| | l2_cache_kb | L2缓存大小 (KB) | 2048 |
| | memory_bandwidth_gbps | 内存带宽 (GB/s) | 512 |
| **约束** | k_alignment | K维度对齐要求 | 32 (必须是32的倍数) |
| | supports_dynamic_batch | 是否支持动态batch | false |
| | requires_power_of_2 | Batch大小必须是2的幂 | true |

### 2.2 硬件特性清单

使用以下清单评估您的硬件：

**✅ 计算特性:**
- [ ] 最大并行tile数量
- [ ] Tensor Core单元数
- [ ] 支持的数据类型 (fp32, fp16, int8)
- [ ] 每个tile的峰值性能 (FLOPS)

**✅ 内存特性:**
- [ ] 缓存层级和大小
- [ ] 内存带宽
- [ ] 是否有本地内存（Scratchpad）

**✅ 约束条件:**
- [ ] Tile维度限制
- [ ] K维度对齐要求
- [ ] Batch大小限制
- [ ] 是否支持不规则矩阵

**✅ 编程模型:**
- [ ] 指令集架构
- [ ] 编译器工具链
- [ ] Runtime API

### 2.3 建立性能模型

**基础性能模型:**

```python
def estimate_execution_time(batch_config, hardware):
    """估算执行时间"""
    # 计算时间
    total_ops = sum(tile.ops for tile in batch_config.tiles)
    compute_time = total_ops / hardware.peak_throughput

    # 内存传输时间
    total_data = sum(tile.data_size for tile in batch_config.tiles)
    memory_time = total_data / hardware.memory_bandwidth

    # 总时间（考虑重叠）
    overlap_factor = 0.8  # 80%的内存传输可与计算重叠
    total_time = compute_time + memory_time * (1 - overlap_factor)

    return total_time
```

**示例:**

```python
# SynapseAI性能模型
synapse_hw = Hardware(
    peak_throughput=8 * 2e12,  # 8 cores × 2 TFLOPS = 16 TFLOPS
    memory_bandwidth=512e9      # 512 GB/s
)

batch = Batch([
    Tile(M=32, K=32, N=32, data_type='fp16')  # 32个这样的tiles
    for _ in range(32)
])

exec_time = estimate_execution_time(batch, synapse_hw)
# 预计: ~40 microseconds
```

---

## 3. 创建硬件配置文件

### 3.1 YAML配置格式

创建 `synapse_ai.yaml`:

```yaml
# SynapseAI硬件配置
hardware:
  name: "synapse_ai"
  version: "1.0"
  description: "SynapseAI Neural Network Accelerator"

compute:
  # 计算能力
  num_cores: 8
  max_batch_size: 32
  tile_size:
    M: 32
    K: 32
    N: 32

  # 支持的数据类型
  supported_dtypes:
    - fp32
    - fp16
    - bf16

  # 峰值性能
  peak_performance:
    fp32_tflops: 16.0
    fp16_tflops: 32.0
    bf16_tflops: 32.0

memory:
  # 缓存层次
  l1_cache_kb: 256
  l2_cache_kb: 2048

  # 带宽
  l1_bandwidth_gbps: 2048
  l2_bandwidth_gbps: 1024
  dram_bandwidth_gbps: 512

  # Tile数据大小（fp16）
  tile_data_size_kb: 6  # (32*32 + 32*32 + 32*32) * 2 bytes / 1024

constraints:
  # K维度对齐
  k_dimension_alignment: 32
  k_must_be_multiple_of: 32

  # Batch约束
  batch_size_must_be_power_of_2: true
  min_batch_size: 8
  max_batch_size: 32

  # Tile约束
  tile_m_range: [16, 64]
  tile_k_range: [32, 64]  # 必须>=32
  tile_n_range: [16, 64]

  # 其他约束
  supports_dynamic_batching: false
  requires_k_reduction_in_software: true
  max_k_iterations: 64

scheduling:
  # 调度策略
  prefer_k_grouping: true       # 优先将相同K维度的tiles分组
  allow_cross_matrix: true      # 允许跨矩阵batching
  prioritize_cache_reuse: true  # 优先考虑缓存复用

  # 启发式权重
  weights:
    k_alignment_penalty: 100.0    # K对齐失败的惩罚
    cache_miss_penalty: 50.0      # Cache miss的惩罚
    nop_penalty: 10.0             # NOP的惩罚
    load_imbalance_penalty: 20.0  # 负载不均衡的惩罚

performance_model:
  # 基准性能数据
  ops_per_tile_fp16: 65536      # 32*32*32*2
  cycles_per_tile: 512          # 实测

  # 延迟
  batch_launch_overhead_ns: 500
  tile_launch_overhead_ns: 50

  # 内存延迟
  l1_hit_latency_ns: 10
  l2_hit_latency_ns: 50
  dram_latency_ns: 200

backend:
  # 后端代码生成
  target_dialect: "synapse"     # 自定义MLIR Dialect
  codegen_strategy: "tiled_loop"

  # 编译器选项
  compiler_flags:
    - "-O3"
    - "-march=synapse-v1"
    - "-ffast-math"

  # Runtime库
  runtime_library: "libsynapse_runtime.so"

  # 内置函数
  intrinsics:
    matmul_tile: "synapse_tc_matmul_tile"
    batch_execute: "synapse_batch_execute"
    accumulate: "synapse_accumulate_fp16"
```

### 3.2 配置文件验证

**创建验证脚本 `validate_config.py`:**

```python
import yaml
from typing import Dict, Any

class HardwareConfigValidator:
    REQUIRED_FIELDS = {
        'hardware': ['name', 'version'],
        'compute': ['num_cores', 'max_batch_size', 'tile_size'],
        'memory': ['l1_cache_kb', 'dram_bandwidth_gbps'],
        'constraints': [],
        'scheduling': ['weights'],
        'backend': ['target_dialect']
    }

    @staticmethod
    def validate(config_path: str) -> bool:
        """验证配置文件"""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        # 检查必需字段
        for section, required_keys in HardwareConfigValidator.REQUIRED_FIELDS.items():
            if section not in config:
                print(f"❌ Missing section: {section}")
                return False

            for key in required_keys:
                if key not in config[section]:
                    print(f"❌ Missing key: {section}.{key}")
                    return False

        # 检查tile_size格式
        tile_size = config['compute']['tile_size']
        if not all(k in tile_size for k in ['M', 'K', 'N']):
            print("❌ tile_size must contain M, K, N")
            return False

        # 检查batch_size合理性
        max_batch = config['compute']['max_batch_size']
        if max_batch <= 0 or max_batch > 1024:
            print(f"❌ Unreasonable max_batch_size: {max_batch}")
            return False

        # 检查约束一致性
        if 'constraints' in config:
            constraints = config['constraints']
            if 'batch_size_must_be_power_of_2' in constraints:
                if constraints['batch_size_must_be_power_of_2']:
                    import math
                    if not (max_batch & (max_batch - 1)) == 0:
                        print(f"❌ max_batch_size={max_batch} not power of 2")
                        return False

        print("✅ Configuration valid")
        return True

# 使用
if __name__ == "__main__":
    HardwareConfigValidator.validate("synapse_ai.yaml")
```

---

## 4. Python优化器集成

### 4.1 加载硬件配置

**扩展 `python/optimizer/hardware_config.py`:**

```python
from dataclasses import dataclass
from typing import List, Optional
import yaml

@dataclass
class HardwareConfig:
    """硬件配置类"""
    name: str
    num_cores: int
    max_batch_size: int
    tile_size: tuple[int, int, int]

    # 约束
    k_alignment: int = 1
    k_must_be_multiple: Optional[int] = None
    batch_must_be_power_of_2: bool = False
    min_batch_size: int = 1

    # 性能模型
    ops_per_tile: int = 0
    peak_throughput: float = 0.0
    memory_bandwidth: float = 0.0

    # 调度权重
    k_alignment_penalty: float = 100.0
    cache_miss_penalty: float = 50.0
    nop_penalty: float = 10.0

    @classmethod
    def from_yaml(cls, config_path: str) -> 'HardwareConfig':
        """从YAML文件加载配置"""
        with open(config_path) as f:
            config = yaml.safe_load(f)

        compute = config['compute']
        constraints = config.get('constraints', {})
        scheduling = config.get('scheduling', {})
        weights = scheduling.get('weights', {})
        perf_model = config.get('performance_model', {})

        return cls(
            name=config['hardware']['name'],
            num_cores=compute['num_cores'],
            max_batch_size=compute['max_batch_size'],
            tile_size=(
                compute['tile_size']['M'],
                compute['tile_size']['K'],
                compute['tile_size']['N']
            ),
            k_alignment=constraints.get('k_dimension_alignment', 1),
            k_must_be_multiple=constraints.get('k_must_be_multiple_of'),
            batch_must_be_power_of_2=constraints.get('batch_size_must_be_power_of_2', False),
            min_batch_size=constraints.get('min_batch_size', 1),
            ops_per_tile=perf_model.get('ops_per_tile_fp16', 0),
            k_alignment_penalty=weights.get('k_alignment_penalty', 100.0),
            cache_miss_penalty=weights.get('cache_miss_penalty', 50.0),
            nop_penalty=weights.get('nop_penalty', 10.0),
        )

# 使用示例
synapse_config = HardwareConfig.from_yaml("configs/synapse_ai.yaml")
print(f"Loaded config for {synapse_config.name}")
print(f"Max batch size: {synapse_config.max_batch_size}")
print(f"Tile size: {synapse_config.tile_size}")
```

### 4.2 适配约束求解器

**扩展 `python/optimizer/cpsat_optimizer.py`:**

```python
from ortools.sat.python import cp_model

class CPSATOptimizerWithCustomHardware(CPSATOptimizer):
    def __init__(self, hw_config: HardwareConfig):
        super().__init__(hw_config)
        self.hw_config = hw_config

    def build_constraints(self, model: cp_model.CpModel, tiles, batch_vars):
        """构建约束，包括自定义硬件约束"""
        # 基本约束
        super().build_constraints(model, tiles, batch_vars)

        # 自定义约束1: K维度必须对齐
        if self.hw_config.k_must_be_multiple:
            for i, tile in enumerate(tiles):
                if tile.K % self.hw_config.k_must_be_multiple != 0:
                    # 添加高惩罚
                    penalty = model.NewIntVar(0, 10000, f"k_align_penalty_{i}")
                    model.Add(penalty == self.hw_config.k_alignment_penalty)
                    # 加入目标函数

        # 自定义约束2: Batch大小必须是2的幂
        if self.hw_config.batch_must_be_power_of_2:
            valid_batch_sizes = [2**i for i in range(10)
                                if 2**i <= self.hw_config.max_batch_size]

            for batch_id in range(self.num_batches):
                batch_size = model.NewIntVar(0, self.hw_config.max_batch_size,
                                            f"batch_size_{batch_id}")
                # 限制batch_size只能取有效值
                model.AddAllowedAssignments([batch_size],
                                           [[s] for s in valid_batch_sizes])

        # 自定义约束3: 最小batch大小
        for batch_id in range(self.num_batches):
            tiles_in_batch = [batch_vars[i][batch_id] for i in range(len(tiles))]
            batch_size = sum(tiles_in_batch)

            # 如果batch非空，必须>=min_batch_size
            model.Add(batch_size == 0).OnlyEnforceIf(
                model.NewBoolVar(f"batch_{batch_id}_empty")
            )
            model.Add(batch_size >= self.hw_config.min_batch_size).OnlyEnforceIf(
                model.NewBoolVar(f"batch_{batch_id}_nonempty")
            )

    def compute_objective(self, model, tiles, batch_vars):
        """计算目标函数，使用自定义权重"""
        objective_terms = []

        # 1. NOP惩罚
        for batch_id in range(self.num_batches):
            tiles_in_batch = sum(batch_vars[i][batch_id] for i in range(len(tiles)))
            nops = self.hw_config.max_batch_size - tiles_in_batch
            nop_cost = model.NewIntVar(0, 100000, f"nop_cost_{batch_id}")
            model.Add(nop_cost == nops * int(self.hw_config.nop_penalty))
            objective_terms.append(nop_cost)

        # 2. K对齐惩罚
        # ... (如上所述)

        # 3. Cache miss惩罚
        for batch_id in range(self.num_batches):
            # 估算cache miss
            cache_misses = self.estimate_cache_misses(tiles, batch_vars, batch_id)
            cache_cost = model.NewIntVar(0, 100000, f"cache_cost_{batch_id}")
            model.Add(cache_cost == cache_misses * int(self.hw_config.cache_miss_penalty))
            objective_terms.append(cache_cost)

        # 最小化总成本
        model.Minimize(sum(objective_terms))
```

### 4.3 性能模型集成

```python
class SynapseAIPerformanceModel:
    """SynapseAI性能模型"""

    def __init__(self, hw_config: HardwareConfig):
        self.hw_config = hw_config
        self.peak_throughput = 32e12  # 32 TFLOPS (fp16)
        self.memory_bandwidth = 512e9  # 512 GB/s

    def estimate_batch_time(self, batch: List[Tile]) -> float:
        """估算batch执行时间（秒）"""
        # 计算时间
        total_ops = sum(tile.M * tile.K * tile.N * 2 for tile in batch)
        compute_time = total_ops / self.peak_throughput

        # 内存时间
        total_data_bytes = sum(
            (tile.M * tile.K + tile.K * tile.N + tile.M * tile.N) * 2  # fp16
            for tile in batch
        )
        memory_time = total_data_bytes / self.memory_bandwidth

        # 启动开销
        launch_overhead = 500e-9  # 500ns

        # 考虑80%计算-内存重叠
        overlap_factor = 0.8
        total_time = max(compute_time, memory_time * (1 - overlap_factor)) + launch_overhead

        return total_time

    def estimate_total_time(self, schedule: List[List[Tile]]) -> float:
        """估算总执行时间"""
        return sum(self.estimate_batch_time(batch) for batch in schedule)

# 使用
perf_model = SynapseAIPerformanceModel(synapse_config)
schedule = optimizer.optimize(tiles)
total_time = perf_model.estimate_total_time(schedule)
print(f"Estimated execution time: {total_time * 1e6:.2f} µs")
```

---

## 5. MLIR Dialect扩展

### 5.1 定义SynapseAI Dialect

**创建 `include/Dialects/Synapse/SynapseDialect.td`:**

```tablegen
// SynapseAI Dialect定义

#ifndef SYNAPSE_DIALECT
#define SYNAPSE_DIALECT

include "mlir/IR/OpBase.td"

def Synapse_Dialect : Dialect {
  let name = "synapse";
  let summary = "SynapseAI Neural Network Accelerator Dialect";
  let description = [{
    This dialect provides operations for the SynapseAI accelerator.
  }];
  let cppNamespace = "::mlir::synapse";
}

// Base class for Synapse operations
class Synapse_Op<string mnemonic, list<Trait> traits = []> :
    Op<Synapse_Dialect, mnemonic, traits>;

#endif // SYNAPSE_DIALECT
```

**创建 `include/Dialects/Synapse/SynapseOps.td`:**

```tablegen
// SynapseAI Operations

#ifndef SYNAPSE_OPS
#define SYNAPSE_OPS

include "SynapseDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Tensor Core Batch操作
def Synapse_TCBatchOp : Synapse_Op<"tc_batch", [Pure]> {
  let summary = "SynapseAI Tensor Core batch matmul";
  let description = [{
    Executes a batch of tile matmuls on SynapseAI Tensor Cores.

    Example:
    ```mlir
    %result = synapse.tc_batch %tiles_a, %tiles_b {
      batch_size = 32 : i32,
      tile_size = [32, 32, 32]
    } : (tensor<32x32x32xf16>, tensor<32x32x32xf16>) -> tensor<32x32x32xf16>
    ```
  }];

  let arguments = (ins
    AnyTensor:$tiles_a,
    AnyTensor:$tiles_b,
    I32Attr:$batch_size,
    I64ArrayAttr:$tile_size
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $tiles_a `,` $tiles_b attr-dict `:` functional-type(operands, results)
  }];
}

// 累加操作
def Synapse_AccumulateOp : Synapse_Op<"accumulate", [Pure]> {
  let summary = "Accumulate fp16 results";
  let description = [{
    Accumulates partial results using SynapseAI's hardware accumulator.
  }];

  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
}

// DMA传输操作
def Synapse_DMATransferOp : Synapse_Op<"dma_transfer"> {
  let summary = "DMA transfer between memory levels";

  let arguments = (ins
    AnyTensor:$input,
    StrAttr:$src_memory,  // "dram", "l2", "l1"
    StrAttr:$dst_memory
  );

  let results = (outs AnyTensor:$result);
}

#endif // SYNAPSE_OPS
```

### 5.2 BatchComp到SynapseAI转换

**创建 `lib/Dialects/Synapse/BatchCompToSynapse.cpp`:**

```cpp
#include "mlir/Pass/Pass.h"
#include "BatchComp/BatchCompOps.h"
#include "Synapse/SynapseOps.h"

namespace mlir {
namespace synapse {

// ExecuteBatchOp → SynapseAI tc_batch转换
class ExecuteBatchToSynapse : public OpRewritePattern<batch_comp::ExecuteBatchOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(batch_comp::ExecuteBatchOp op,
                                PatternRewriter &rewriter) const override {
    // 只处理hardware="synapse"的操作
    if (op->getAttrOfType<StringAttr>("hardware").getValue() != "synapse")
      return failure();

    // 提取tile数据
    Value tilesA = extractTiles(op, 'A');
    Value tilesB = extractTiles(op, 'B');

    // 创建SynapseAI tc_batch操作
    auto batchSize = rewriter.getI32IntegerAttr(32);  // 从配置读取
    auto tileSize = rewriter.getI64ArrayAttr({32, 32, 32});

    auto synapseOp = rewriter.create<TCBatchOp>(
      op.getLoc(),
      op.getType(),
      tilesA,
      tilesB,
      batchSize,
      tileSize
    );

    rewriter.replaceOp(op, synapseOp.getResult());
    return success();
  }

private:
  Value extractTiles(batch_comp::ExecuteBatchOp op, char matrix) const {
    // 从schedule中提取tiles...
    // (实际实现需要访问schedule数据结构)
  }
};

// AccumulateOp转换
class AccumulateToSynapse : public OpRewritePattern<batch_comp::AccumulateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(batch_comp::AccumulateOp op,
                                PatternRewriter &rewriter) const override {
    // 创建SynapseAI accumulate操作
    auto synapseOp = rewriter.create<AccumulateOp>(
      op.getLoc(),
      op.getType(),
      op.getLhs(),
      op.getRhs()
    );

    rewriter.replaceOp(op, synapseOp.getResult());
    return success();
  }
};

// Pass定义
class BatchCompToSynapsePass
    : public PassWrapper<BatchCompToSynapsePass, OperationPass<ModuleOp>> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    // 添加转换模式
    patterns.add<ExecuteBatchToSynapse>(context);
    patterns.add<AccumulateToSynapse>(context);

    // 应用转换
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace synapse
} // namespace mlir
```

---

## 6. 后端代码生成

### 6.1 从SynapseAI Dialect到LLVM IR

**创建 `lib/Dialects/Synapse/SynapseToLLVM.cpp`:**

```cpp
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "Synapse/SynapseOps.h"

namespace mlir {
namespace synapse {

// tc_batch → runtime调用
class TCBatchToLLVM : public ConvertOpToLLVMPattern<TCBatchOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      TCBatchOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    MLIRContext *context = op.getContext();

    // 准备参数
    Value tilesA = adaptor.getTilesA();
    Value tilesB = adaptor.getTilesB();
    Value batchSize = rewriter.create<LLVM::ConstantOp>(
      loc, rewriter.getI32Type(), op.getBatchSizeAttr()
    );

    // 声明runtime函数: synapse_tc_batch_execute
    auto funcType = LLVM::LLVMFunctionType::get(
      getVoidPtrType(),  // 返回结果指针
      {
        getVoidPtrType(),  // tiles_a
        getVoidPtrType(),  // tiles_b
        rewriter.getI32Type()  // batch_size
      }
    );

    LLVM::LLVMFuncOp runtimeFunc = lookupOrCreateFn(
      op->getParentOfType<ModuleOp>(),
      "synapse_tc_batch_execute",
      funcType
    );

    // 调用runtime函数
    auto callOp = rewriter.create<LLVM::CallOp>(
      loc,
      runtimeFunc,
      ValueRange{tilesA, tilesB, batchSize}
    );

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

// accumulate → runtime调用
class AccumulateToLLVM : public ConvertOpToLLVMPattern<AccumulateOp> {
public:
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult matchAndRewrite(
      AccumulateOp op,
      OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    // 声明runtime函数
    auto funcType = LLVM::LLVMFunctionType::get(
      getVoidPtrType(),
      {getVoidPtrType(), getVoidPtrType()}
    );

    LLVM::LLVMFuncOp runtimeFunc = lookupOrCreateFn(
      op->getParentOfType<ModuleOp>(),
      "synapse_accumulate_fp16",
      funcType
    );

    // 调用
    auto callOp = rewriter.create<LLVM::CallOp>(
      loc,
      runtimeFunc,
      ValueRange{adaptor.getLhs(), adaptor.getRhs()}
    );

    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

} // namespace synapse
} // namespace mlir
```

### 6.2 Runtime库实现

**创建 `runtime/synapse_runtime.cpp`:**

```cpp
// SynapseAI Runtime库

#include <cstdint>
#include <cstring>

extern "C" {

// Tensor Core Batch执行
void* synapse_tc_batch_execute(
    void* tiles_a,
    void* tiles_b,
    int32_t batch_size
) {
    // 1. 将数据加载到L1 Cache
    synapse_dma_transfer(tiles_a, DRAM, L1_CACHE, batch_size * TILE_SIZE_BYTES);
    synapse_dma_transfer(tiles_b, DRAM, L1_CACHE, batch_size * TILE_SIZE_BYTES);

    // 2. 配置Tensor Core
    SynapseTCConfig config;
    config.batch_size = batch_size;
    config.tile_m = 32;
    config.tile_k = 32;
    config.tile_n = 32;
    config.data_type = FP16;

    synapse_tc_configure(&config);

    // 3. 启动批处理
    void* result = synapse_tc_launch(tiles_a, tiles_b, &config);

    // 4. 等待完成
    synapse_tc_wait();

    return result;
}

// FP16累加
void* synapse_accumulate_fp16(void* lhs, void* rhs) {
    // 使用硬件累加器
    void* result = synapse_allocate(TENSOR_SIZE);
    synapse_acc_add_fp16(result, lhs, rhs);
    return result;
}

// DMA传输
void synapse_dma_transfer(
    void* data,
    MemoryLevel src,
    MemoryLevel dst,
    size_t bytes
) {
    // 配置DMA
    SynapseDMAConfig dma;
    dma.src_addr = data;
    dma.dst_level = dst;
    dma.size_bytes = bytes;

    // 启动DMA
    synapse_dma_start(&dma);
    synapse_dma_wait();
}

} // extern "C"
```

---

## 7. 测试和验证

### 7.1 单元测试

**创建 `test/Dialects/Synapse/batch-comp-to-synapse.mlir`:**

```mlir
// RUN: mlir-opt %s --batch-comp-to-synapse | FileCheck %s

module {
  func.func @test_batch_execute(%tiles_a: !batch_comp.tileset, %tiles_b: !batch_comp.tileset) -> tensor<32x32xf16> {
    // CHECK-LABEL: func.func @test_batch_execute
    // CHECK-NOT: batch_comp.execute_batch
    // CHECK: synapse.tc_batch
    // CHECK-SAME: batch_size = 32
    // CHECK-SAME: tile_size = [32, 32, 32]
    %result = batch_comp.execute_batch %tiles_a, %tiles_b {
      hardware = "synapse",
      batch_size = 32 : i32
    } : (!batch_comp.tileset, !batch_comp.tileset) -> tensor<32x32xf16>

    return %result : tensor<32x32xf16>
  }
}
```

### 7.2 端到端测试

**创建 `test/Integration/synapse_e2e.mlir`:**

```mlir
// RUN: batch-comp-opt %s \
// RUN:   --batch-comp-tile-scheduler="hardware=synapse config-file=synapse_ai.yaml" \
// RUN:   --batch-comp-lowering \
// RUN:   --batch-comp-to-synapse \
// RUN:   --synapse-to-llvm \
// RUN:   --reconcile-unrealized-casts \
// RUN: | FileCheck %s

module {
  func.func @matmul_64x64(%A: tensor<64x64xf16>, %B: tensor<64x64xf16>) -> tensor<64x64xf16> {
    // CHECK: llvm.call @synapse_tc_batch_execute
    %C = tosa.matmul %A, %B : (tensor<64x64xf16>, tensor<64x64xf16>) -> tensor<64x64xf16>
    return %C : tensor<64x64xf16>
  }
}
```

### 7.3 性能基准测试

**创建 `benchmarks/synapse_benchmark.cpp`:**

```cpp
#include <benchmark/benchmark.h>
#include "synapse_runtime.h"

static void BM_SynapseTCBatch(benchmark::State& state) {
  int batch_size = state.range(0);

  // 准备数据
  void* tiles_a = allocate_tiles(batch_size);
  void* tiles_b = allocate_tiles(batch_size);

  for (auto _ : state) {
    void* result = synapse_tc_batch_execute(tiles_a, tiles_b, batch_size);
    benchmark::DoNotOptimize(result);
  }

  state.SetItemsProcessed(state.iterations() * batch_size);
}

BENCHMARK(BM_SynapseTCBatch)->Range(8, 32);

BENCHMARK_MAIN();
```

---

## 8. 性能调优

### 8.1 Profiling

```python
from python.optimizer import CPSATOptimizer
from python.profiler import HardwareProfiler

# 创建profiler
profiler = HardwareProfiler(hardware="synapse")

# 运行优化并profiling
with profiler:
    optimizer = CPSATOptimizer(synapse_config)
    schedule = optimizer.optimize(tiles)

# 分析结果
report = profiler.generate_report()
print(f"Total execution time: {report.total_time_us} µs")
print(f"Compute utilization: {report.compute_util:.1f}%")
print(f"Memory bandwidth utilization: {report.memory_util:.1f}%")
print(f"Average batch size: {report.avg_batch_size:.1f}")
```

### 8.2 调优参数

**调整权重以优化性能:**

```yaml
# synapse_ai_tuned.yaml

scheduling:
  weights:
    # 原始权重
    k_alignment_penalty: 100.0
    nop_penalty: 10.0
    cache_miss_penalty: 50.0

    # 调优后的权重（通过实验确定）
    k_alignment_penalty: 150.0  # ↑ 更重视K对齐
    nop_penalty: 5.0            # ↓ 允许更多NOP以获得更好对齐
    cache_miss_penalty: 80.0    # ↑ 更重视cache复用
```

### 8.3 自动调优

```python
import optuna

def objective(trial):
    """目标函数：最小化执行时间"""
    # 采样超参数
    k_penalty = trial.suggest_float("k_alignment_penalty", 50, 200)
    nop_penalty = trial.suggest_float("nop_penalty", 1, 20)
    cache_penalty = trial.suggest_float("cache_miss_penalty", 20, 100)

    # 创建配置
    config = synapse_config.copy()
    config.k_alignment_penalty = k_penalty
    config.nop_penalty = nop_penalty
    config.cache_miss_penalty = cache_penalty

    # 运行优化
    optimizer = CPSATOptimizer(config)
    schedule = optimizer.optimize(test_tiles)

    # 测量实际性能
    exec_time = benchmark_schedule(schedule)

    return exec_time

# 运行Optuna优化
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(f"Best parameters: {study.best_params}")
print(f"Best execution time: {study.best_value} µs")
```

---

## 9. 完整示例

### 9.1 完整工作流程

**从TOSA到SynapseAI可执行代码:**

```bash
#!/bin/bash
# build_for_synapse.sh

# 步骤1: TOSA matmul → BatchComp IR
mlir-opt input.mlir \
  --batch-comp-tile-scheduler="hardware=synapse config-file=configs/synapse_ai.yaml algorithm=cpsat" \
  -o step1_batchcomp.mlir

# 步骤2: 高层BatchComp → 低层BatchComp
mlir-opt step1_batchcomp.mlir \
  --batch-comp-lowering \
  -o step2_lowered.mlir

# 步骤3: BatchComp → SynapseAI Dialect
mlir-opt step2_lowered.mlir \
  --batch-comp-to-synapse \
  -o step3_synapse.mlir

# 步骤4: SynapseAI → LLVM IR
mlir-opt step3_synapse.mlir \
  --synapse-to-llvm \
  --reconcile-unrealized-casts \
  -o step4_llvm.mlir

# 步骤5: LLVM IR → 目标代码
mlir-translate step4_llvm.mlir \
  --mlir-to-llvmir \
  -o output.ll

# 步骤6: 链接runtime库
clang output.ll -lsynapse_runtime -o executable

echo "✅ Build complete: executable"
```

### 9.2 运行和验证

```bash
# 运行可执行文件
./executable

# 使用SynapseAI模拟器验证
synapse-sim --trace executable

# 性能分析
synapse-prof executable
```

---

## 10. 总结

### 10.1 检查清单

完成以下步骤后，您的自定义硬件应该已完全集成：

- [ ] 创建硬件配置文件 (YAML)
- [ ] 扩展Python优化器支持自定义约束
- [ ] 定义MLIR Dialect和操作
- [ ] 实现BatchComp到自定义Dialect的转换Pass
- [ ] 实现自定义Dialect到LLVM的转换Pass
- [ ] 编写Runtime库
- [ ] 创建单元测试
- [ ] 创建端到端测试
- [ ] 进行性能基准测试和调优

### 10.2 常见问题

**Q: 如何处理不支持的约束？**

A: 在CP-SAT优化器中添加软约束（带惩罚的约束），而不是硬约束。

**Q: Runtime性能不如预期怎么办？**

A: 使用profiler找出瓶颈（计算 vs 内存），然后调整调度权重或优化runtime实现。

**Q: 如何支持多种数据类型？**

A: 在YAML配置中列出`supported_dtypes`，在Dialect中使用类型参数。

### 10.3 参考资料

- [MLIR Dialect教程](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)
- [CP-SAT求解器文档](https://developers.google.com/optimization/cp/cp_solver)
- [BatchComp架构文档](../design/ARCHITECTURE.md)

---

**恭喜！** 您现在应该能够为BatchComp框架添加自定义硬件支持了。
