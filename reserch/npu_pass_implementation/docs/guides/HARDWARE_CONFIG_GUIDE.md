# Hardware Configuration Guide

**如何为不同硬件配置BatchComp框架**

本指南详细介绍如何创建、修改和优化BatchComp的硬件配置，以充分发挥不同加速器的性能。

---

## 📋 目录

1. [配置文件格式](#1-配置文件格式)
2. [内置硬件配置](#2-内置硬件配置)
3. [创建自定义配置](#3-创建自定义配置)
4. [配置参数详解](#4-配置参数详解)
5. [性能调优指南](#5-性能调优指南)
6. [常见问题](#6-常见问题)

---

## 1. 配置文件格式

### 1.1 YAML格式（推荐）

```yaml
# hardware_config.yaml

hardware:
  name: "MyNPU"
  type: "npu"               # npu, gpu, tpu, custom
  vendor: "MyCompany"       # 可选

compute:
  batch_size: 16            # 每批的tile数量
  tile_size: [16, 16, 16]   # Tile维度 [M, K, N]
  must_fill_batch: true     # 是否必须填满batch
  supports_variable_tiles: true
  supports_mixed_precision: false
  supported_dtypes:
    - float32
    - float16
  compute_throughput: 0.5   # TFLOPS

memory:
  on_chip_buffer: 16384     # 片上缓存大小(字节)
  l2_cache: null            # L2缓存（可选）
  ddr_bandwidth: 50.0       # DDR带宽 (GB/s)
  dma_channels: 2
  double_buffer: true       # 双缓冲支持

optimization:
  default_algorithm: "auto" # auto, greedy, improved_greedy, cpsat
  cp_sat_time_limit: 10.0   # CP-SAT超时(秒)
  cp_sat_enabled_threshold: 100
  greedy_threshold: 1000
  enable_multi_matmul_fusion: true
  enable_k_reduction_grouping: true

vendor_extensions:
  # 厂商特定扩展（可选）
  custom_feature: true
```

### 1.2 JSON格式

```json
{
  "hardware": {
    "name": "MyNPU",
    "type": "npu"
  },
  "compute": {
    "batch_size": 16,
    "tile_size": [16, 16, 16],
    "must_fill_batch": true
  },
  "memory": {
    "on_chip_buffer": 16384,
    "ddr_bandwidth": 50.0
  },
  "optimization": {
    "default_algorithm": "auto"
  }
}
```

### 1.3 Python代码

```python
from python.core.hardware_config import (
    HardwareConfig, HardwareType,
    ComputeConfig, MemoryConfig, OptimizationConfig
)

config = HardwareConfig(
    name="MyNPU",
    hardware_type=HardwareType.NPU,
    compute=ComputeConfig(
        batch_size=16,
        tile_size=(16, 16, 16),
        must_fill_batch=True
    ),
    memory=MemoryConfig(
        on_chip_size=16384,
        ddr_bandwidth=50.0
    ),
    optimization=OptimizationConfig(
        default_algorithm="auto"
    )
)
```

---

## 2. 内置硬件配置

### 2.1 NPU配置

```python
config = HardwareConfig.get_default("npu")
```

**参数:**
- Batch Size: 16
- Tile Size: 16×16×16
- Must Fill Batch: True
- Compute: 0.5 TFLOPS
- Memory: 16KB on-chip, 50 GB/s DDR

**适用场景:**
- 边缘AI推理
- 低功耗场景
- 严格批次对齐要求

### 2.2 GPU配置

```python
config = HardwareConfig.get_default("gpu")
```

**参数:**
- Batch Size: 32 (Warp size)
- Tile Size: 16×16×16 (Tensor Core)
- Must Fill Batch: False
- Compute: 10 TFLOPS
- Memory: 160KB shared memory, 40MB L2, 900 GB/s HBM

**适用场景:**
- 云端训练/推理
- 高性能计算
- 支持混合精度

### 2.3 TPU配置

```python
config = HardwareConfig.get_default("tpu")
```

**参数:**
- Batch Size: 128
- Tile Size: 128×128×128
- Must Fill Batch: False
- Compute: 100 TFLOPS
- Memory: 16MB vector memory, 1200 GB/s HBM

**适用场景:**
- 大规模模型训练
- 批量推理
- Google Cloud环境

---

## 3. 创建自定义配置

### 3.1 从现有配置修改

```python
# 基于NPU配置修改
config = HardwareConfig.get_default("npu")

# 修改batch size
config.compute.batch_size = 24

# 修改tile size
config.compute.tile_size = (32, 32, 32)

# 启用混合精度
config.compute.supports_mixed_precision = True
config.compute.supported_dtypes = ["float32", "float16", "bfloat16"]

# 调整优化策略
config.optimization.default_algorithm = "cpsat"
config.optimization.cp_sat_time_limit = 20.0

# 保存
config.to_yaml("custom_npu.yaml")
```

### 3.2 完全自定义

```python
config = HardwareConfig(
    name="AI-Chip-X100",
    hardware_type=HardwareType.CUSTOM,
    compute=ComputeConfig(
        batch_size=20,                    # 自定义batch大小
        tile_size=(24, 24, 24),           # 非标准tile尺寸
        must_fill_batch=False,            # 可以部分填充
        supports_variable_tiles=True,     # 支持可变尺寸
        supports_mixed_precision=True,
        supported_dtypes=["float32", "bfloat16", "int8"],
        compute_throughput=3.5            # 3.5 TFLOPS
    ),
    memory=MemoryConfig(
        on_chip_size=65536,               # 64KB
        l2_cache_size=2097152,            # 2MB L2
        ddr_bandwidth=200.0,              # 200 GB/s
        dma_channels=4,
        supports_double_buffering=True
    ),
    optimization=OptimizationConfig(
        default_algorithm="cpsat",
        cp_sat_time_limit=30.0,
        cp_sat_enabled_threshold=500,
        greedy_threshold=5000,
        enable_multi_matmul_fusion=True,
        enable_k_reduction_grouping=True
    ),
    vendor_extensions={
        'vendor': 'MyChipCompany',
        'chip_version': 'X100',
        'special_instructions': ['matmul_fp16', 'conv_int8']
    }
)

# 验证配置
if config.validate():
    print("✓ Configuration valid")
    config.to_yaml("ai_chip_x100.yaml")
else:
    print("✗ Invalid configuration")
```

### 3.3 从硬件规格表创建

假设您有硬件规格表：

| 参数 | 值 |
|------|-----|
| Tensor Core数量 | 8 |
| 每核心可并行tiles | 4 |
| Tile维度 | 32×32×32 |
| On-chip SRAM | 128KB |
| DDR带宽 | 100 GB/s |
| 峰值性能 | 8 TFLOPS |

**创建配置:**

```python
config = HardwareConfig(
    name="Example-Chip",
    hardware_type=HardwareType.CUSTOM,
    compute=ComputeConfig(
        batch_size=8 * 4,              # 8核×4tiles = 32
        tile_size=(32, 32, 32),
        must_fill_batch=False,
        compute_throughput=8.0
    ),
    memory=MemoryConfig(
        on_chip_size=128 * 1024,       # 128KB
        ddr_bandwidth=100.0
    ),
    optimization=OptimizationConfig(
        default_algorithm="auto"
    )
)
```

---

## 4. 配置参数详解

### 4.1 Hardware Section

| 参数 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `name` | string | 配置名称 | "NPU-16" |
| `type` | enum | 硬件类型 | "npu", "gpu", "tpu", "custom" |
| `vendor` | string | 厂商名称（可选） | "Nvidia", "Google" |

### 4.2 Compute Section

#### batch_size

**描述**: 单次可并行执行的tile数量

**如何确定**:
```
batch_size = 计算核心数 × 每核心可并行tiles数
```

**示例:**
- NPU: 1核 × 16 tiles = 16
- GPU: 32 warps × 1 tile = 32
- TPU: 128 matrix units × 1 tile = 128

**注意事项:**
- 必须 > 0
- 推荐设为2的幂（8, 16, 32, 64, 128）
- 过大会增加调度开销
- 过小会降低硬件利用率

#### tile_size

**描述**: 单个tile的维度 `[M, K, N]`

**如何确定:**
```
tile_size = 硬件Tensor Core支持的最大尺寸
```

**常见值:**
- NPU: [16, 16, 16]
- GPU Tensor Core: [16, 16, 16]
- TPU MXU: [128, 128, 128]
- Custom: 根据硬件手册

**权衡:**
- 更大tile → 减少tile数量，降低调度开销
- 更小tile → 更好的负载均衡，减少边缘浪费

#### must_fill_batch

**描述**: batch是否必须完全填满

**设置建议:**
- `true`: 硬件强制要求（如某些SIMD架构）
- `false`: 硬件允许部分batch（大多数现代加速器）

**影响:**
- `true` → 可能增加NOP数量
- `false` → 更灵活的调度

#### supports_variable_tiles

**描述**: 是否支持可变尺寸的tiles

**设置建议:**
- `true`: 硬件可以处理边缘tiles（不满16×16×16）
- `false`: 必须padding到标准尺寸

#### supported_dtypes

**描述**: 支持的数据类型列表

**常见值:**
```yaml
supported_dtypes:
  - float32       # FP32 (所有硬件)
  - float16       # FP16 (Tensor Core, TPU)
  - bfloat16      # BF16 (TPU, 新GPU)
  - int8          # INT8 (推理加速)
```

### 4.3 Memory Section

#### on_chip_buffer

**描述**: 片上缓存大小（字节）

**如何确定**: 查阅硬件手册的SRAM/Scratchpad大小

**示例:**
- NPU: 16KB - 256KB
- GPU Shared Memory: 48KB - 164KB per SM
- TPU Vector Memory: 16MB

#### ddr_bandwidth

**描述**: DDR/HBM带宽（GB/s）

**如何确定**: 查阅硬件规格

**示例:**
- DDR4: 20-50 GB/s
- GDDR6: 400-900 GB/s
- HBM2: 900-1200 GB/s

**用途**: 用于性能建模和调度优化

### 4.4 Optimization Section

#### default_algorithm

**描述**: 默认优化算法

**选项:**
- `"auto"`: 自动选择（推荐）
- `"greedy"`: 贪心算法
- `"improved_greedy"`: 改进贪心
- `"cpsat"`: CP-SAT约束求解

**建议:**
- 开发调试: `"greedy"` (最快)
- 生产环境: `"auto"` (平衡)
- 性能关键: `"cpsat"` (最优)

#### cp_sat_time_limit

**描述**: CP-SAT求解器超时时间（秒）

**建议值:**
- 快速编译: 5.0
- 标准: 10.0
- 离线优化: 30.0 - 300.0

#### 阈值参数

```yaml
cp_sat_enabled_threshold: 100    # ≤ 100 tiles启用CP-SAT
greedy_threshold: 1000           # > 1000 tiles使用greedy
```

**调整指南:**
| Tile数量 | 推荐算法 | 阈值设置 |
|----------|----------|----------|
| < 50 | ImprovedGreedy | - |
| 50-500 | CP-SAT | cp_sat_enabled_threshold=500 |
| 500-2000 | ImprovedGreedy | - |
| > 2000 | Greedy | greedy_threshold=2000 |

---

## 5. 性能调优指南

### 5.1 调优流程

```
1. 使用默认配置baseline测试
   ↓
2. Profiling找出瓶颈
   ↓
3. 调整相关参数
   ↓
4. 对比性能提升
   ↓
5. 迭代优化
```

### 5.2 常见调优场景

#### 场景1: 编译时间过长

**症状**: CP-SAT求解超时

**解决方案:**
```yaml
optimization:
  cp_sat_time_limit: 5.0          # 缩短超时
  cp_sat_enabled_threshold: 50    # 降低阈值
  default_algorithm: "improved_greedy"  # 改用启发式
```

#### 场景2: Runtime性能不佳

**症状**: 过多NOP，utilization低

**解决方案:**
```yaml
optimization:
  default_algorithm: "cpsat"      # 使用最优算法
  cp_sat_time_limit: 30.0         # 增加求解时间
  enable_multi_matmul_fusion: true  # 启用跨矩阵优化
```

#### 场景3: 内存带宽瓶颈

**症状**: 计算单元空闲，等待数据

**解决方案:**
```yaml
memory:
  double_buffer: true              # 启用双缓冲
  dma_channels: 4                  # 增加DMA通道

optimization:
  enable_k_reduction_grouping: true  # 减少内存访问
```

### 5.3 参数调优示例

```python
import time
from python.core.hardware_config import HardwareConfig
from python.core.batch_optimizer import UnifiedOptimizer
from python.core.tile_info import generate_tiles_for_matmul

# 基准配置
base_config = HardwareConfig.get_default("npu")
tiles = generate_tiles_for_matmul(m=128, n=128, k=128)

# 测试不同配置
configs = {
    "Baseline": base_config,

    "Fast Compile": HardwareConfig.get_default("npu"),
    "Best Performance": HardwareConfig.get_default("npu"),
}

# 调整配置
configs["Fast Compile"].optimization.default_algorithm = "greedy"
configs["Best Performance"].optimization.default_algorithm = "cpsat"
configs["Best Performance"].optimization.cp_sat_time_limit = 60.0

# 运行对比
print(f"{'Config':<20} {'Compile(ms)':<15} {'Batches':<10} {'NOPs':<10} {'Util%'}")
print("-" * 70)

for name, config in configs.items():
    optimizer = UnifiedOptimizer(config)

    start = time.time()
    result = optimizer.optimize(tiles)
    compile_time = (time.time() - start) * 1000

    print(f"{name:<20} {compile_time:<15.2f} {result.num_batches:<10} "
          f"{result.total_nops:<10} {result.utilization*100:.1f}")
```

---

## 6. 常见问题

### Q1: 如何选择batch_size?

**A**: 根据硬件能力:
```
batch_size = 硬件并行单元数 × SIMD宽度
```

例如:
- 16个Tensor Core → batch_size=16
- 32个CUDA Warp → batch_size=32

### Q2: tile_size必须是16×16×16吗?

**A**: 不一定。根据硬件支持:
- Tensor Core通常是16×16×16
- 某些NPU支持32×32×32
- TPU可以是128×128×128

### Q3: must_fill_batch应该设为true还是false?

**A**:
- SIMD架构（所有lane必须执行）→ `true`
- MIMD架构（独立执行单元）→ `false`
- 不确定时设为`false`（更灵活）

### Q4: 如何验证配置正确性?

**A**:
```python
config = HardwareConfig.from_yaml("my_config.yaml")

# 验证
if config.validate():
    print("✓ Valid")
else:
    print("✗ Invalid")

# 测试运行
optimizer = UnifiedOptimizer(config)
tiles = generate_tiles_for_matmul(64, 64, 64)
result = optimizer.optimize(tiles)
print(f"Test successful: {result}")
```

### Q5: 配置文件可以热更新吗?

**A**: 可以，重新加载即可:
```python
# 修改配置文件后
config = HardwareConfig.from_yaml("updated_config.yaml")
optimizer = UnifiedOptimizer(config)
```

---

## 7. 完整配置示例

### 示例1: 移动端NPU

```yaml
# mobile_npu.yaml
hardware:
  name: "Mobile-NPU-V2"
  type: "npu"
  vendor: "Qualcomm"

compute:
  batch_size: 8
  tile_size: [16, 16, 16]
  must_fill_batch: true
  supports_variable_tiles: false
  supported_dtypes: ["float16", "int8"]
  compute_throughput: 0.3

memory:
  on_chip_buffer: 8192
  ddr_bandwidth: 20.0
  dma_channels: 1
  double_buffer: false

optimization:
  default_algorithm: "greedy"      # 快速编译
  enable_multi_matmul_fusion: false  # 简化调度
```

### 示例2: 数据中心GPU

```yaml
# datacenter_gpu.yaml
hardware:
  name: "A100-80GB"
  type: "gpu"
  vendor: "Nvidia"

compute:
  batch_size: 32
  tile_size: [16, 16, 16]
  must_fill_batch: false
  supports_variable_tiles: true
  supports_mixed_precision: true
  supported_dtypes: ["float32", "float16", "bfloat16", "int8"]
  compute_throughput: 312.0

memory:
  on_chip_buffer: 163840          # 160KB per SM
  l2_cache: 41943040              # 40MB
  ddr_bandwidth: 2000.0           # HBM2e
  dma_channels: 1
  double_buffer: true

optimization:
  default_algorithm: "cpsat"
  cp_sat_time_limit: 10.0
  cp_sat_enabled_threshold: 1000
  greedy_threshold: 5000
  enable_multi_matmul_fusion: true
  enable_k_reduction_grouping: false
```

---

**更新日期**: 2025-11-15
