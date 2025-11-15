"""
Batch Composition Optimization Framework

统一的批组合优化框架，支持多种硬件平台（NPU、GPU、TPU）。
"""

__version__ = "0.1.0"

from .core.tile_info import TileInfo
from .core.hardware_config import (
    HardwareConfig,
    HardwareType,
    ComputeConfig,
    MemoryConfig,
    OptimizationConfig,
)
from .core.batch_optimizer import (
    BatchOptimizer,
    BatchScheduleResult,
    UnifiedOptimizer,
)

# 导出常用类
__all__ = [
    # 核心数据结构
    "TileInfo",
    "HardwareConfig",
    "HardwareType",
    "ComputeConfig",
    "MemoryConfig",
    "OptimizationConfig",
    # 优化器
    "BatchOptimizer",
    "BatchScheduleResult",
    "UnifiedOptimizer",
]
