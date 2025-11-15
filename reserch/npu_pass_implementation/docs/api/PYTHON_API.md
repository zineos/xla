# Python API Reference

**BatchComp Python优化器完整API文档**

本文档提供BatchComp Python模块的完整API参考，包括所有类、函数、参数和返回值。

---

## 📋 目录

1. [核心数据结构](#1-核心数据结构)
2. [硬件配置](#2-硬件配置)
3. [优化器基类](#3-优化器基类)
4. [优化算法](#4-优化算法)
5. [辅助工具](#5-辅助工具)
6. [C++接口](#6-c接口)

---

## 1. 核心数据结构

### 1.1 TileInfo

**位置**: `python/core/tile_info.py`

表示单个tile计算的完整信息。

```python
@dataclass
class TileInfo:
    # 唯一标识符
    tile_id: int                      # Tile的唯一ID
    matrix_id: int                    # 所属矩阵的ID（多矩阵场景）

    # 在计算中的位置
    m_offset: int                     # 输出矩阵的行偏移
    n_offset: int                     # 输出矩阵的列偏移
    k_iteration: int                  # K维度迭代索引（用于规约）

    # 实际维度（可能小于tile_size）
    actual_m: int                     # 实际M维度
    actual_n: int                     # 实际N维度
    actual_k: int                     # 实际K维度

    # 源矩阵的切片信息
    a_slice: Tuple[int, int, int, int]  # [m_start, m_end, k_start, k_end]
    b_slice: Tuple[int, int, int, int]  # [k_start, k_end, n_start, n_end]
    c_slice: Tuple[int, int, int, int]  # [m_start, m_end, n_start, n_end]

    # 计算属性
    accumulate: bool = False          # 是否累加到已有C值
    tile_type: TileType = TileType.COMPUTE  # Tile类型

    # 优化提示
    priority: int = 0                 # 优先级（越高越先调度）
    dependencies: List[int] = None    # 依赖的tile_id列表
```

**属性（只读）:**

```python
@property
def compute_volume(self) -> int:
    """计算量（MAC操作数）"""
    return self.actual_m * self.actual_n * self.actual_k

@property
def padding_ratio(self) -> float:
    """填充比例（0.0 = 无填充, 1.0 = 完全填充）"""
    max_volume = 16 * 16 * 16
    return 1.0 - (self.compute_volume / max_volume)

@property
def group_key(self) -> Tuple[int, int]:
    """分组键，用于约束强制: (matrix_id, k_iteration)"""
    return (self.matrix_id, self.k_iteration)

@property
def is_edge_tile(self) -> bool:
    """是否为边缘tile（非满16×16×16）"""
    return (self.actual_m < 16 or
            self.actual_n < 16 or
            self.actual_k < 16)
```

**方法:**

```python
def create_nop(self) -> 'TileInfo':
    """创建一个NOP tile用于填充"""
```

**示例:**

```python
from python.core.tile_info import TileInfo, TileType

# 创建一个tile
tile = TileInfo(
    tile_id=0,
    matrix_id=0,
    m_offset=0,
    n_offset=0,
    k_iteration=0,
    actual_m=16,
    actual_n=16,
    actual_k=16,
    a_slice=(0, 16, 0, 16),
    b_slice=(0, 16, 0, 16),
    c_slice=(0, 16, 0, 16),
    accumulate=False,
    tile_type=TileType.COMPUTE
)

print(f"Compute volume: {tile.compute_volume}")  # 4096
print(f"Is edge tile: {tile.is_edge_tile}")      # False
```

---

### 1.2 TileType

**位置**: `python/core/tile_info.py`

Tile类型枚举。

```python
class TileType(Enum):
    COMPUTE = "compute"   # 计算tile
    NOP = "nop"           # 空操作（填充）
    PADDING = "padding"   # 填充tile
```

---

### 1.3 BatchScheduleResult

**位置**: `python/core/batch_optimizer.py`

批调度结果的完整信息。

```python
@dataclass
class BatchScheduleResult:
    batches: List[List[TileInfo]]   # 每个批次包含的tiles
    num_batches: int                # 批次数量
    total_nops: int                 # 总NOP数量
    utilization: float              # 平均利用率 (0.0-1.0)
    algorithm: str                  # 使用的算法名称
    execution_time: float           # 执行时间（秒）
    metadata: Dict[str, Any]        # 额外元数据
```

**方法:**

```python
def __str__(self) -> str:
    """格式化字符串表示"""
```

**示例:**

```python
result = optimizer.optimize(tiles)

print(f"Batches: {result.num_batches}")
print(f"NOPs: {result.total_nops}")
print(f"Utilization: {result.utilization:.2%}")
print(f"Algorithm: {result.algorithm}")
print(f"Time: {result.execution_time:.4f}s")

# 访问元数据
print(f"Total tiles: {result.metadata['total_tiles']}")
```

---

### 1.4 辅助函数

#### generate_tiles_for_matmul()

**位置**: `python/core/tile_info.py`

为单个矩阵乘法生成tiles。

```python
def generate_tiles_for_matmul(
    m: int,
    n: int,
    k: int,
    matrix_id: int = 0,
    tile_size: Tuple[int, int, int] = (16, 16, 16)
) -> List[TileInfo]:
    """
    生成单个矩阵乘法的所有tiles

    参数:
        m, n, k: 矩阵维度 (M×K @ K×N = M×N)
        matrix_id: 矩阵的唯一标识符
        tile_size: Tile维度 (tile_m, tile_k, tile_n)

    返回:
        TileInfo列表
    """
```

**示例:**

```python
from python.core.tile_info import generate_tiles_for_matmul

# 64×64 @ 64×64 矩阵乘法，使用16×16×16 tiles
tiles = generate_tiles_for_matmul(m=64, n=64, k=64, matrix_id=0)

print(f"Total tiles: {len(tiles)}")  # 64个tiles
# (64/16) × (64/16) × (64/16) = 4 × 4 × 4 = 64
```

#### generate_tiles_for_multiple_matmuls()

为多个矩阵乘法生成tiles。

```python
def generate_tiles_for_multiple_matmuls(
    matmul_specs: List[dict]
) -> List[TileInfo]:
    """
    生成多个矩阵乘法的tiles

    参数:
        matmul_specs: 矩阵规格列表，每个dict包含'm', 'n', 'k'键

    返回:
        合并的TileInfo列表，tile_id和matrix_id全局唯一
    """
```

**示例:**

```python
specs = [
    {'m': 64, 'n': 64, 'k': 64},    # Matrix 0
    {'m': 32, 'n': 32, 'k': 32},    # Matrix 1
    {'m': 48, 'n': 48, 'k': 48}     # Matrix 2
]

tiles = generate_tiles_for_multiple_matmuls(specs)

# 检查矩阵ID
matrix_ids = set(t.matrix_id for t in tiles)
print(f"Matrix IDs: {matrix_ids}")  # {0, 1, 2}

# Tile ID全局唯一
tile_ids = [t.tile_id for t in tiles]
print(f"Unique tile IDs: {len(set(tile_ids)) == len(tile_ids)}")  # True
```

---

## 2. 硬件配置

### 2.1 HardwareConfig

**位置**: `python/core/hardware_config.py`

完整的硬件配置类。

```python
@dataclass
class HardwareConfig:
    name: str                          # 配置名称
    hardware_type: HardwareType        # 硬件类型
    compute: ComputeConfig             # 计算配置
    memory: MemoryConfig               # 内存配置
    optimization: OptimizationConfig   # 优化配置
    vendor_extensions: Dict[str, Any]  # 厂商扩展字段
```

**类方法:**

```python
@classmethod
def from_yaml(cls, yaml_path: str) -> 'HardwareConfig':
    """从YAML文件加载配置"""

@classmethod
def from_json(cls, json_path: str) -> 'HardwareConfig':
    """从JSON文件加载配置"""

@classmethod
def get_default(cls, hardware: str) -> 'HardwareConfig':
    """
    获取默认硬件配置

    参数:
        hardware: "npu", "gpu", 或 "tpu"

    返回:
        相应的默认配置
    """

@classmethod
def default_npu(cls) -> 'HardwareConfig':
    """NPU默认配置"""

@classmethod
def default_gpu(cls) -> 'HardwareConfig':
    """GPU默认配置"""

@classmethod
def default_tpu(cls) -> 'HardwareConfig':
    """TPU默认配置"""
```

**实例方法:**

```python
def to_yaml(self, yaml_path: str) -> None:
    """保存配置到YAML文件"""

def to_json(self, json_path: str) -> None:
    """保存配置到JSON文件"""

def validate(self) -> bool:
    """验证配置一致性"""
```

**示例:**

```python
from python.core.hardware_config import HardwareConfig

# 方法1: 使用默认配置
npu_config = HardwareConfig.get_default("npu")
gpu_config = HardwareConfig.get_default("gpu")

# 方法2: 从文件加载
custom_config = HardwareConfig.from_yaml("my_hardware.yaml")

# 方法3: 手动创建
from python.core.hardware_config import (
    HardwareType, ComputeConfig, MemoryConfig, OptimizationConfig
)

config = HardwareConfig(
    name="Custom-NPU",
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

# 保存配置
config.to_yaml("custom_npu.yaml")
```

---

### 2.2 HardwareType

硬件类型枚举。

```python
class HardwareType(Enum):
    NPU = "npu"
    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"
    CUSTOM = "custom"
```

---

### 2.3 ComputeConfig

计算能力配置。

```python
@dataclass
class ComputeConfig:
    batch_size: int                       # 每批的tile数量
    tile_size: Tuple[int, int, int]       # Tile维度 (M, N, K)
    must_fill_batch: bool                 # 是否必须填满batch
    supports_variable_tiles: bool = False # 是否支持可变tile大小
    supports_mixed_precision: bool = False# 是否支持混合精度
    supported_dtypes: list = ["float32"]  # 支持的数据类型
    compute_throughput: float = 1.0       # 计算吞吐量(TFLOPS)
```

---

### 2.4 MemoryConfig

内存层次配置。

```python
@dataclass
class MemoryConfig:
    on_chip_size: int                         # 片上缓存大小(字节)
    l2_cache_size: Optional[int] = None       # L2缓存大小
    ddr_bandwidth: float = 100.0              # DDR带宽(GB/s)
    dma_channels: int = 1                     # DMA通道数
    supports_double_buffering: bool = False   # 是否支持双缓冲
```

---

### 2.5 OptimizationConfig

优化策略配置。

```python
@dataclass
class OptimizationConfig:
    default_algorithm: str = "auto"             # 默认算法
    cp_sat_time_limit: float = 10.0            # CP-SAT超时时间(秒)
    cp_sat_enabled_threshold: int = 50         # 启用CP-SAT的tile数阈值
    greedy_threshold: int = 1000               # 使用贪心的tile数阈值
    enable_multi_matmul_fusion: bool = True    # 启用多矩阵融合
    enable_k_reduction_grouping: bool = True   # 启用K维规约分组
```

---

## 3. 优化器基类

### 3.1 BatchOptimizer

**位置**: `python/core/batch_optimizer.py`

所有优化算法的抽象基类。

```python
class BatchOptimizer(ABC):
    def __init__(self, config: HardwareConfig):
        """
        初始化优化器

        参数:
            config: 硬件配置
        """
```

**抽象方法（子类必须实现）:**

```python
@abstractmethod
def optimize(
    self,
    tiles: List[TileInfo],
    **kwargs
) -> BatchScheduleResult:
    """
    优化tile到batch的分配

    参数:
        tiles: 待调度的tiles列表
        **kwargs: 算法特定的参数

    返回:
        BatchScheduleResult: 调度结果
    """

@abstractmethod
def get_algorithm_name(self) -> str:
    """返回算法名称"""
```

**工具方法:**

```python
def validate_tiles(self, tiles: List[TileInfo]) -> None:
    """
    验证tiles的有效性

    异常:
        ValueError: tiles无效时抛出
    """

def calculate_utilization(self, batches: List[List[TileInfo]]) -> float:
    """计算批次的平均利用率 (0.0-1.0)"""

def count_nops(self, batches: List[List[TileInfo]]) -> int:
    """计算总NOP数量"""

def group_tiles_by_matrix(
    self,
    tiles: List[TileInfo]
) -> Dict[int, List[TileInfo]]:
    """按matrix_id分组tiles"""

def group_tiles_by_k_iteration(
    self,
    tiles: List[TileInfo]
) -> Dict[int, List[TileInfo]]:
    """按k_iteration分组tiles"""
```

---

## 4. 优化算法

### 4.1 GreedyOptimizer

**位置**: `python/optimizers/greedy.py`

简单贪心调度算法。

```python
class GreedyOptimizer(BatchOptimizer):
    """
    贪心批调度算法

    算法: 按tile_id顺序填充批次，批次满了开始新批次

    时间复杂度: O(n log n)
    空间复杂度: O(n)

    适用场景:
    - 单矩阵
    - 快速编译
    - Tile数量 > 10000
    """

    def __init__(self, config: HardwareConfig):
        """
        初始化贪心优化器

        参数:
            config: 硬件配置
        """
```

**方法:**

```python
def optimize(
    self,
    tiles: List[TileInfo],
    **kwargs
) -> BatchScheduleResult:
    """
    使用贪心算法调度tiles

    参数:
        tiles: 待调度的tiles
        **kwargs: 未使用

    返回:
        BatchScheduleResult
    """

def get_algorithm_name(self) -> str:
    """返回 "greedy" """
```

**示例:**

```python
from python.optimizers.greedy import GreedyOptimizer
from python.core.hardware_config import HardwareConfig
from python.core.tile_info import generate_tiles_for_matmul

# 创建配置和tiles
config = HardwareConfig.get_default("npu")
tiles = generate_tiles_for_matmul(m=64, n=64, k=64)

# 优化
optimizer = GreedyOptimizer(config)
result = optimizer.optimize(tiles)

print(f"Batches: {result.num_batches}")
print(f"Utilization: {result.utilization:.2%}")
```

---

### 4.2 ImprovedGreedyOptimizer

**位置**: `python/optimizers/improved_greedy.py`

改进贪心算法，支持K维分组和跨矩阵优化。

```python
class ImprovedGreedyOptimizer(BatchOptimizer):
    """
    改进贪心批调度算法

    改进:
    - K维规约分组
    - 跨矩阵填充优化
    - 大组优先处理

    时间复杂度: O(n log n)
    空间复杂度: O(n)

    适用场景:
    - 多矩阵优化
    - 100-10000 tiles
    - **推荐作为默认算法**
    """

    def __init__(self, config: HardwareConfig):
        """
        初始化改进贪心优化器

        参数:
            config: 硬件配置
        """
```

**方法:**

```python
def optimize(
    self,
    tiles: List[TileInfo],
    **kwargs
) -> BatchScheduleResult:
    """
    使用改进贪心算法调度tiles

    参数:
        tiles: 待调度的tiles
        **kwargs: 未使用

    返回:
        BatchScheduleResult
    """

def get_algorithm_name(self) -> str:
    """返回 "improved_greedy" """
```

**示例:**

```python
from python.optimizers.improved_greedy import ImprovedGreedyOptimizer
from python.core.hardware_config import HardwareConfig
from python.core.tile_info import generate_tiles_for_multiple_matmuls

# 多矩阵场景
specs = [
    {'m': 64, 'n': 64, 'k': 64},
    {'m': 32, 'n': 32, 'k': 32},
    {'m': 48, 'n': 48, 'k': 48}
]
tiles = generate_tiles_for_multiple_matmuls(specs)

config = HardwareConfig.get_default("npu")
optimizer = ImprovedGreedyOptimizer(config)
result = optimizer.optimize(tiles)

print(f"Matrices: {result.metadata['num_matrices']}")
print(f"Groups: {result.metadata['num_groups']}")
print(f"NOPs: {result.total_nops}")
```

---

### 4.3 CPSATOptimizer

**位置**: `python/optimizers/cpsat_optimizer.py`

基于Google OR-Tools CP-SAT的约束规划求解器。

```python
class CPSATOptimizer(BatchOptimizer):
    """
    CP-SAT约束求解批调度算法

    特点:
    - 全局最优解（给定足够时间）
    - 多目标优化
    - 支持复杂约束

    时间复杂度: NP (有时间限制)
    空间复杂度: O(n * m) (n=tiles, m=batches)

    适用场景:
    - 高性能要求
    - Tile数量 < 5000
    - 可接受较长编译时间
    - **推荐用于生产环境**

    依赖:
        需要安装: pip install ortools
    """

    def __init__(
        self,
        config: HardwareConfig,
        time_limit: Optional[float] = None
    ):
        """
        初始化CP-SAT优化器

        参数:
            config: 硬件配置
            time_limit: CP-SAT求解器超时时间(秒)，
                       None则使用配置中的值

        异常:
            ImportError: 如果ortools未安装
        """
```

**方法:**

```python
def optimize(
    self,
    tiles: List[TileInfo],
    **kwargs
) -> BatchScheduleResult:
    """
    使用CP-SAT求解器优化tile调度

    参数:
        tiles: 待调度的tiles
        **kwargs: 额外参数
            - time_limit (float): 覆盖默认超时时间

    返回:
        BatchScheduleResult

    注意:
        如果求解失败，自动回退到ImprovedGreedyOptimizer
    """

def get_algorithm_name(self) -> str:
    """返回 "cpsat" """
```

**示例:**

```python
from python.optimizers.cpsat_optimizer import CPSATOptimizer
from python.core.hardware_config import HardwareConfig
from python.core.tile_info import generate_tiles_for_multiple_matmuls

specs = [
    {'m': 64, 'n': 64, 'k': 64},
    {'m': 48, 'n': 48, 'k': 48}
]
tiles = generate_tiles_for_multiple_matmuls(specs)

config = HardwareConfig.get_default("npu")

# 创建CP-SAT优化器，30秒超时
optimizer = CPSATOptimizer(config, time_limit=30.0)
result = optimizer.optimize(tiles)

print(f"Algorithm: {result.algorithm}")
print(f"Batches: {result.num_batches}")
print(f"NOPs: {result.total_nops}")
print(f"Time: {result.execution_time:.2f}s")
```

---

### 4.4 UnifiedOptimizer

**位置**: `python/core/batch_optimizer.py`

统一优化器，根据场景自动选择最佳算法。

```python
class UnifiedOptimizer:
    """
    统一优化器 - 自动选择最佳算法

    选择策略:
    - 单矩阵 → Greedy (最快)
    - 多矩阵 + tiles < 50 → ImprovedGreedy
    - 多矩阵 + 50 ≤ tiles ≤ 1000 → CP-SAT (最优)
    - 多矩阵 + tiles > 1000 → ImprovedGreedy (平衡)
    """

    def __init__(self, config: HardwareConfig):
        """
        初始化统一优化器

        参数:
            config: 硬件配置
        """
```

**方法:**

```python
def optimize(
    self,
    tiles: List[TileInfo],
    algorithm: str = "auto",
    **kwargs
) -> BatchScheduleResult:
    """
    优化tile调度

    参数:
        tiles: tiles列表
        algorithm: 算法选择
            - "auto": 自动选择（默认）
            - "greedy": 强制使用贪心
            - "improved_greedy": 强制使用改进贪心
            - "cpsat": 强制使用CP-SAT
        **kwargs: 传递给具体算法的参数

    返回:
        BatchScheduleResult
    """
```

**示例:**

```python
from python.core.batch_optimizer import UnifiedOptimizer
from python.core.hardware_config import HardwareConfig
from python.core.tile_info import generate_tiles_for_multiple_matmuls

specs = [
    {'m': 64, 'n': 64, 'k': 64},
    {'m': 32, 'n': 32, 'k': 32}
]
tiles = generate_tiles_for_multiple_matmuls(specs)

config = HardwareConfig.get_default("npu")
optimizer = UnifiedOptimizer(config)

# 方法1: 自动选择算法
result_auto = optimizer.optimize(tiles, algorithm="auto")
print(f"Auto selected: {result_auto.algorithm}")

# 方法2: 强制使用特定算法
result_cpsat = optimizer.optimize(tiles, algorithm="cpsat", time_limit=30.0)
print(f"Forced CP-SAT: {result_cpsat.algorithm}")

# 方法3: 算法对比
for algo in ["greedy", "improved_greedy", "cpsat"]:
    result = optimizer.optimize(tiles, algorithm=algo)
    print(f"{algo:20} batches={result.num_batches:3} nops={result.total_nops:4} util={result.utilization:.1%}")
```

---

## 5. 辅助工具

### 5.1 C++接口（cpp_interface.py）

**位置**: `python/interface/cpp_interface.py`

提供Python和C++ MLIR Passes之间的接口。

```python
def initialize_interface(hardware_config: dict) -> None:
    """
    初始化Python-C++接口

    参数:
        hardware_config: 硬件配置字典
    """

def optimize_tiles(
    tiles: List[dict],
    algorithm: str = "auto",
    hardware: str = "npu",
    **kwargs
) -> dict:
    """
    从C++调用Python优化器

    参数:
        tiles: Tile信息字典列表
        algorithm: 算法名称
        hardware: 硬件类型
        **kwargs: 额外参数

    返回:
        调度结果字典，包含:
        - batches: 批次列表
        - num_batches: 批次数量
        - total_nops: NOP数量
        - utilization: 利用率
    """
```

**示例（从C++调用）:**

```python
# C++通过pybind11调用

import python.interface.cpp_interface as interface

# 1. 初始化
interface.initialize_interface({
    'hardware': 'npu',
    'batch_size': 16
})

# 2. 准备tiles数据
tiles_data = [
    {
        'tile_id': 0,
        'matrix_id': 0,
        'M': 16, 'K': 16, 'N': 16,
        'k_iteration': 0
    },
    # ... 更多tiles
]

# 3. 调用优化器
result = interface.optimize_tiles(
    tiles=tiles_data,
    algorithm="cpsat",
    hardware="npu"
)

print(f"Batches: {result['num_batches']}")
print(f"NOPs: {result['total_nops']}")
```

---

## 6. 完整示例

### 6.1 基本使用流程

```python
# 1. 导入模块
from python.core.tile_info import generate_tiles_for_matmul
from python.core.hardware_config import HardwareConfig
from python.core.batch_optimizer import UnifiedOptimizer

# 2. 生成tiles
tiles = generate_tiles_for_matmul(m=64, n=64, k=64)
print(f"Generated {len(tiles)} tiles")

# 3. 创建硬件配置
config = HardwareConfig.get_default("npu")

# 4. 创建优化器
optimizer = UnifiedOptimizer(config)

# 5. 优化
result = optimizer.optimize(tiles, algorithm="auto")

# 6. 分析结果
print(f"Algorithm: {result.algorithm}")
print(f"Batches: {result.num_batches}")
print(f"NOPs: {result.total_nops}")
print(f"Utilization: {result.utilization:.2%}")
print(f"Time: {result.execution_time:.4f}s")

# 7. 访问批次详情
for i, batch in enumerate(result.batches):
    print(f"Batch {i}: {len(batch)} tiles")
    for tile in batch:
        print(f"  Tile {tile.tile_id}: matrix={tile.matrix_id}, K={tile.k_iteration}")
```

### 6.2 多矩阵优化

```python
from python.core.tile_info import generate_tiles_for_multiple_matmuls
from python.core.hardware_config import HardwareConfig
from python.optimizers.cpsat_optimizer import CPSATOptimizer

# 定义多个矩阵
specs = [
    {'m': 100, 'n': 200, 'k': 150},  # Matrix 0
    {'m': 50, 'n': 50, 'k': 50},     # Matrix 1
    {'m': 75, 'n': 100, 'k': 80}     # Matrix 2
]

# 生成所有tiles
tiles = generate_tiles_for_multiple_matmuls(specs)
print(f"Total tiles: {len(tiles)}")

# 按矩阵统计
from collections import Counter
matrix_counts = Counter(t.matrix_id for t in tiles)
for mat_id, count in sorted(matrix_counts.items()):
    print(f"Matrix {mat_id}: {count} tiles")

# 使用CP-SAT优化（最优解）
config = HardwareConfig.get_default("npu")
optimizer = CPSATOptimizer(config, time_limit=60.0)
result = optimizer.optimize(tiles)

# 分析跨矩阵batching
for i, batch in enumerate(result.batches):
    matrices_in_batch = set(t.matrix_id for t in batch)
    k_values = set(t.k_iteration for t in batch)
    print(f"Batch {i}: {len(batch)} tiles from matrices {matrices_in_batch}, K={k_values}")
```

### 6.3 自定义硬件配置

```python
from python.core.hardware_config import (
    HardwareConfig, HardwareType,
    ComputeConfig, MemoryConfig, OptimizationConfig
)
from python.core.tile_info import generate_tiles_for_matmul
from python.core.batch_optimizer import UnifiedOptimizer

# 创建自定义硬件配置
custom_config = HardwareConfig(
    name="Custom-AI-Chip",
    hardware_type=HardwareType.CUSTOM,
    compute=ComputeConfig(
        batch_size=24,              # 自定义batch大小
        tile_size=(32, 32, 32),     # 自定义tile大小
        must_fill_batch=False,
        supports_variable_tiles=True,
        compute_throughput=2.5      # 2.5 TFLOPS
    ),
    memory=MemoryConfig(
        on_chip_size=32768,         # 32KB
        ddr_bandwidth=100.0,
        dma_channels=4,
        supports_double_buffering=True
    ),
    optimization=OptimizationConfig(
        default_algorithm="cpsat",
        cp_sat_time_limit=20.0,
        cp_sat_enabled_threshold=200,
        greedy_threshold=2000,
        enable_multi_matmul_fusion=True,
        enable_k_reduction_grouping=True
    ),
    vendor_extensions={
        'vendor': 'MyCompany',
        'chip_version': 'v2.0',
        'custom_feature': True
    }
)

# 保存配置
custom_config.to_yaml("custom_chip.yaml")

# 验证配置
if custom_config.validate():
    print("Configuration valid!")

# 使用自定义配置
tiles = generate_tiles_for_matmul(m=128, n=128, k=128)
optimizer = UnifiedOptimizer(custom_config)
result = optimizer.optimize(tiles)

print(f"Custom hardware result: {result}")
```

### 6.4 性能对比

```python
from python.core.tile_info import generate_tiles_for_multiple_matmuls
from python.core.hardware_config import HardwareConfig
from python.optimizers import (
    GreedyOptimizer,
    ImprovedGreedyOptimizer,
    CPSATOptimizer
)
import time

# 准备测试数据
specs = [{'m': 64, 'n': 64, 'k': 64} for _ in range(5)]
tiles = generate_tiles_for_multiple_matmuls(specs)

config = HardwareConfig.get_default("npu")

# 测试所有算法
algorithms = {
    'Greedy': GreedyOptimizer(config),
    'ImprovedGreedy': ImprovedGreedyOptimizer(config),
    'CP-SAT': CPSATOptimizer(config, time_limit=30.0)
}

print(f"{'Algorithm':<20} {'Batches':<10} {'NOPs':<10} {'Util%':<10} {'Time(ms)':<12}")
print("-" * 70)

for name, optimizer in algorithms.items():
    result = optimizer.optimize(tiles)
    print(f"{name:<20} {result.num_batches:<10} {result.total_nops:<10} "
          f"{result.utilization*100:<10.1f} {result.execution_time*1000:<12.2f}")
```

---

## 7. 异常处理

### 常见异常

```python
# 1. Tiles验证失败
try:
    optimizer.optimize(invalid_tiles)
except ValueError as e:
    print(f"Invalid tiles: {e}")

# 2. CP-SAT未安装
try:
    optimizer = CPSATOptimizer(config)
except ImportError as e:
    print(f"CP-SAT not available: {e}")
    print("Install with: pip install ortools")

# 3. 配置文件不存在
try:
    config = HardwareConfig.from_yaml("nonexistent.yaml")
except FileNotFoundError as e:
    print(f"Config file not found: {e}")

# 4. CP-SAT求解失败（自动fallback）
result = cpsat_optimizer.optimize(tiles)
if result.algorithm == "improved_greedy":
    print("CP-SAT failed, used fallback algorithm")
```

---

## 8. 最佳实践

### 8.1 算法选择

```python
# 推荐：使用UnifiedOptimizer让框架自动选择
optimizer = UnifiedOptimizer(config)
result = optimizer.optimize(tiles)  # 自动选择最佳算法
```

### 8.2 性能调优

```python
# 对于大规模问题，调整CP-SAT超时
config.optimization.cp_sat_time_limit = 5.0  # 缩短超时

# 对于离线优化，可以增加超时获得更好结果
optimizer = CPSATOptimizer(config, time_limit=300.0)  # 5分钟
```

### 8.3 多次运行取最优

```python
best_result = None
best_score = float('inf')

for _ in range(5):  # 运行5次
    result = optimizer.optimize(tiles)
    score = result.num_batches * 1000 + result.total_nops

    if score < best_score:
        best_score = score
        best_result = result

print(f"Best result: {best_result}")
```

---

## 9. 版本兼容性

**当前版本**: BatchComp v1.0 (Phase 3: 85%)

**Python要求**: Python >= 3.8

**依赖**:
- `ortools` >= 9.0 (可选, CP-SAT算法需要)
- `pyyaml` >= 5.0 (配置文件加载)
- `dataclasses` (Python 3.7+内置)

**安装**:
```bash
pip install ortools pyyaml
```

---

**文档更新**: 2025-11-15
