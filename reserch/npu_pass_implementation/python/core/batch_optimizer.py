"""
批组合优化器基类

定义所有优化算法的统一接口。
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from .tile_info import TileInfo
from .hardware_config import HardwareConfig


@dataclass
class BatchScheduleResult:
    """批调度结果"""
    batches: List[List[TileInfo]]  # 每个批次包含的tiles
    num_batches: int               # 批次数量
    total_nops: int                # 总NOP数量
    utilization: float             # 平均利用率
    algorithm: str                 # 使用的算法
    execution_time: float          # 执行时间（秒）
    metadata: Dict[str, Any]       # 额外信息

    def __str__(self) -> str:
        return (
            f"BatchScheduleResult(\n"
            f"  algorithm={self.algorithm},\n"
            f"  batches={self.num_batches},\n"
            f"  nops={self.total_nops},\n"
            f"  utilization={self.utilization:.2%},\n"
            f"  time={self.execution_time:.4f}s\n"
            f")"
        )


class BatchOptimizer(ABC):
    """
    批组合优化器基类

    所有优化算法（贪心、改进贪心、CP-SAT）都继承这个基类。
    """

    def __init__(self, config: HardwareConfig):
        """
        初始化优化器

        Args:
            config: 硬件配置
        """
        self.config = config
        self.batch_size = config.compute.batch_size
        self.must_fill_batch = config.compute.must_fill_batch

    @abstractmethod
    def optimize(
        self,
        tiles: List[TileInfo],
        **kwargs
    ) -> BatchScheduleResult:
        """
        优化tile到batch的分配

        Args:
            tiles: 待调度的tiles列表
            **kwargs: 算法特定的参数

        Returns:
            BatchScheduleResult: 调度结果
        """
        pass

    @abstractmethod
    def get_algorithm_name(self) -> str:
        """返回算法名称"""
        pass

    def validate_tiles(self, tiles: List[TileInfo]) -> None:
        """
        验证tiles的有效性

        Args:
            tiles: 待验证的tiles

        Raises:
            ValueError: 如果tiles无效
        """
        if not tiles:
            raise ValueError("Tiles list cannot be empty")

        # 检查tile_id唯一性
        tile_ids = [t.tile_id for t in tiles]
        if len(tile_ids) != len(set(tile_ids)):
            raise ValueError("Duplicate tile_ids found")

        # 检查所有tiles的大小是否一致（如果硬件要求）
        if not self.config.compute.supports_variable_tiles:
            tile_sizes = set((t.m, t.n, t.k) for t in tiles)
            if len(tile_sizes) > 1:
                raise ValueError(
                    f"Hardware does not support variable tile sizes. "
                    f"Found: {tile_sizes}"
                )

    def calculate_utilization(self, batches: List[List[TileInfo]]) -> float:
        """
        计算批次的平均利用率

        Args:
            batches: 批次列表

        Returns:
            平均利用率 (0.0 - 1.0)
        """
        if not batches:
            return 0.0

        total_utilization = 0.0
        for batch in batches:
            batch_utilization = len(batch) / self.batch_size
            total_utilization += batch_utilization

        return total_utilization / len(batches)

    def count_nops(self, batches: List[List[TileInfo]]) -> int:
        """
        计算总NOP数量

        Args:
            batches: 批次列表

        Returns:
            总NOP数量
        """
        total_nops = 0
        for batch in batches:
            if len(batch) < self.batch_size:
                total_nops += self.batch_size - len(batch)
        return total_nops

    def group_tiles_by_matrix(
        self,
        tiles: List[TileInfo]
    ) -> Dict[int, List[TileInfo]]:
        """
        按matrix_id分组tiles（用于多矩阵场景）

        Args:
            tiles: tiles列表

        Returns:
            {matrix_id: [tiles]}的字典
        """
        groups: Dict[int, List[TileInfo]] = {}
        for tile in tiles:
            if tile.matrix_id not in groups:
                groups[tile.matrix_id] = []
            groups[tile.matrix_id].append(tile)
        return groups

    def group_tiles_by_k_iteration(
        self,
        tiles: List[TileInfo]
    ) -> Dict[int, List[TileInfo]]:
        """
        按k_iteration分组tiles（用于K维规约）

        Args:
            tiles: tiles列表

        Returns:
            {k_iteration: [tiles]}的字典
        """
        groups: Dict[int, List[TileInfo]] = {}
        for tile in tiles:
            if tile.k_iteration not in groups:
                groups[tile.k_iteration] = []
            groups[tile.k_iteration].append(tile)
        return groups


class UnifiedOptimizer:
    """
    统一优化器 - 根据场景自动选择最佳算法

    使用策略：
    - 单矩阵：贪心算法（最快）
    - 多矩阵（<50 tiles）：改进贪心
    - 多矩阵（50-1000 tiles）：CP-SAT（最优）
    - 多矩阵（>1000 tiles）：改进贪心（平衡）
    """

    def __init__(self, config: HardwareConfig):
        self.config = config
        self._optimizers = {}

    def _get_optimizer(self, algorithm: str) -> BatchOptimizer:
        """延迟加载优化器"""
        if algorithm not in self._optimizers:
            if algorithm == "greedy":
                from ..optimizers.greedy import GreedyOptimizer
                self._optimizers[algorithm] = GreedyOptimizer(self.config)
            elif algorithm == "improved_greedy":
                from ..optimizers.improved_greedy import ImprovedGreedyOptimizer
                self._optimizers[algorithm] = ImprovedGreedyOptimizer(self.config)
            elif algorithm == "cpsat":
                from ..optimizers.cpsat_optimizer import CPSATOptimizer
                self._optimizers[algorithm] = CPSATOptimizer(self.config)
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")
        return self._optimizers[algorithm]

    def optimize(
        self,
        tiles: List[TileInfo],
        algorithm: str = "auto",
        **kwargs
    ) -> BatchScheduleResult:
        """
        优化tile调度

        Args:
            tiles: tiles列表
            algorithm: 算法选择 ("auto", "greedy", "improved_greedy", "cpsat")
            **kwargs: 传递给具体算法的参数

        Returns:
            BatchScheduleResult
        """
        if algorithm == "auto":
            algorithm = self._select_algorithm(tiles)

        optimizer = self._get_optimizer(algorithm)
        return optimizer.optimize(tiles, **kwargs)

    def _select_algorithm(self, tiles: List[TileInfo]) -> str:
        """
        自动选择最佳算法

        Args:
            tiles: tiles列表

        Returns:
            算法名称
        """
        # 检查是否有配置的默认算法
        default_algo = self.config.optimization.default_algorithm

        # 统计矩阵数量
        matrix_ids = set(t.matrix_id for t in tiles)
        num_matrices = len(matrix_ids)
        num_tiles = len(tiles)

        # 单矩阵场景：贪心最快
        if num_matrices == 1:
            return "greedy"

        # 多矩阵场景：根据规模选择
        if num_tiles < self.config.optimization.cp_sat_enabled_threshold:
            # 小规模：改进贪心
            return "improved_greedy"
        elif num_tiles <= self.config.optimization.greedy_threshold:
            # 中等规模：CP-SAT最优
            if default_algo == "cp-sat":
                return "cpsat"
            else:
                return "improved_greedy"
        else:
            # 大规模：改进贪心平衡速度和质量
            return "improved_greedy"
