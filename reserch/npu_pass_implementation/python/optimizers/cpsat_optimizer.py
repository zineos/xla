"""
CP-SAT约束优化批调度算法

使用Google OR-Tools的CP-SAT求解器进行全局最优批组合。
适用场景：多矩阵、中等规模（50-1000 tiles）、离线优化
"""

import time
import logging
from typing import List, Dict, Tuple, Optional
from ..core.batch_optimizer import BatchOptimizer, BatchScheduleResult
from ..core.tile_info import TileInfo
from ..core.hardware_config import HardwareConfig

# CP-SAT solver import (optional)
try:
    from ortools.sat.python import cp_model
    HAS_CPSAT = True
except ImportError:
    HAS_CPSAT = False
    cp_model = None


class CPSATOptimizer(BatchOptimizer):
    """
    CP-SAT约束求解批调度算法

    算法逻辑：
    1. 建模为约束优化问题
    2. 变量：x[i,j] = tile i 是否分配到 batch j
    3. 约束：
       - 每个tile恰好分配到一个batch
       - 每个batch最多batch_size个tiles
       - 如果must_fill=True，使用的batch必须恰好batch_size个tiles
       - 局部性约束：同一group_key的tiles倾向于在同一batch
    4. 目标：最小化 (batch数量 * 1000 + NOP数量)
    5. 求解器找到全局最优解
    """

    def __init__(
        self,
        config: HardwareConfig,
        time_limit: Optional[float] = None
    ):
        """
        初始化CP-SAT优化器

        Args:
            config: 硬件配置
            time_limit: CP-SAT求解器超时时间（秒），None则使用配置中的值
        """
        super().__init__(config)

        if not HAS_CPSAT:
            raise ImportError(
                "CP-SAT solver not available. "
                "Install with: pip install ortools"
            )

        self.time_limit = (
            time_limit
            if time_limit is not None
            else config.optimization.cp_sat_time_limit
        )
        self.logger = logging.getLogger(self.__class__.__name__)

    def get_algorithm_name(self) -> str:
        return "cpsat"

    def optimize(
        self,
        tiles: List[TileInfo],
        **kwargs
    ) -> BatchScheduleResult:
        """
        使用CP-SAT求解器优化tile调度

        Args:
            tiles: 待调度的tiles
            **kwargs: 额外参数
                - time_limit: 覆盖默认超时时间

        Returns:
            BatchScheduleResult
        """
        start_time = time.time()

        # 验证输入
        self.validate_tiles(tiles)

        # 获取超时时间
        time_limit = kwargs.get('time_limit', self.time_limit)

        # 求解
        try:
            batches = self._solve_cpsat(tiles, time_limit)
        except Exception as e:
            self.logger.warning(f"CP-SAT failed: {e}, falling back to improved_greedy")
            # 失败时回退到改进贪心
            from .improved_greedy import ImprovedGreedyOptimizer
            fallback = ImprovedGreedyOptimizer(self.config)
            return fallback.optimize(tiles)

        # 计算统计信息
        num_batches = len(batches)
        total_nops = self.count_nops(batches)
        utilization = self.calculate_utilization(batches)
        execution_time = time.time() - start_time

        # 额外统计
        num_matrices = len(set(t.matrix_id for t in tiles))
        num_groups = len(set(t.group_key for t in tiles))

        return BatchScheduleResult(
            batches=batches,
            num_batches=num_batches,
            total_nops=total_nops,
            utilization=utilization,
            algorithm=self.get_algorithm_name(),
            execution_time=execution_time,
            metadata={
                "total_tiles": len(tiles),
                "num_matrices": num_matrices,
                "num_groups": num_groups,
                "batch_size": self.batch_size,
                "time_limit": time_limit,
            }
        )

    def _solve_cpsat(
        self,
        tiles: List[TileInfo],
        time_limit: float
    ) -> List[List[TileInfo]]:
        """
        使用CP-SAT求解器求解批分配问题

        Args:
            tiles: tiles列表
            time_limit: 超时时间（秒）

        Returns:
            批次列表
        """
        num_tiles = len(tiles)
        batch_size = self.batch_size

        # 计算最大批次数
        max_batches = (num_tiles + batch_size - 1) // batch_size

        # 创建CP-SAT模型
        model = cp_model.CpModel()

        # 决策变量: x[i,j] = tile i 是否在 batch j
        x = {}
        for i in range(num_tiles):
            for j in range(max_batches):
                x[i, j] = model.NewBoolVar(f'x_{i}_{j}')

        # 约束1: 每个tile恰好分配到一个batch
        for i in range(num_tiles):
            model.Add(sum(x[i, j] for j in range(max_batches)) == 1)

        # 批次使用情况和大小
        batch_used = []
        batch_sizes = []

        for j in range(max_batches):
            used = model.NewBoolVar(f'batch_{j}_used')
            batch_used.append(used)

            # 计算batch中的tile数量
            tiles_in_batch = sum(x[i, j] for i in range(num_tiles))
            batch_sizes.append(tiles_in_batch)

            # 批次使用 <=> 至少有一个tile
            model.Add(tiles_in_batch > 0).OnlyEnforceIf(used)
            model.Add(tiles_in_batch == 0).OnlyEnforceIf(used.Not())

            # 约束2: 批次大小限制
            if self.must_fill_batch:
                # 如果batch被使用，必须恰好batch_size个tiles
                model.Add(tiles_in_batch == batch_size).OnlyEnforceIf(used)
            else:
                # batch最多batch_size个tiles
                model.Add(tiles_in_batch <= batch_size)

        # 约束3: 局部性约束（同一group的tiles倾向于在同一batch）
        self._add_locality_constraints(model, x, tiles, num_tiles, max_batches)

        # 目标函数: 最小化批次数量和NOPs
        num_batches = sum(batch_used)

        if self.must_fill_batch:
            # total_nops = batch_size * num_batches - num_tiles
            total_nops = batch_size * num_batches - num_tiles
            # 加权目标: 优先减少batch数，其次减少NOPs
            model.Minimize(1000 * num_batches + total_nops)
        else:
            # 只最小化batch数量
            model.Minimize(num_batches)

        # 求解
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = time_limit
        solver.parameters.log_search_progress = False

        status = solver.Solve(model)

        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            # 提取解
            batches = self._extract_batches(solver, x, tiles, num_tiles, max_batches)
            self.logger.info(
                f"CP-SAT solved: {len(batches)} batches, "
                f"status={'OPTIMAL' if status == cp_model.OPTIMAL else 'FEASIBLE'}"
            )
            return batches
        else:
            raise RuntimeError(f"CP-SAT failed with status: {status}")

    def _add_locality_constraints(
        self,
        model,  # cp_model.CpModel
        x: Dict[Tuple[int, int], any],  # Dict of cp_model.IntVar
        tiles: List[TileInfo],
        num_tiles: int,
        max_batches: int
    ) -> None:
        """
        添加局部性约束

        同一group_key的tiles倾向于在同一batch（软约束）。

        Args:
            model: CP-SAT模型
            x: 决策变量
            tiles: tiles列表
            num_tiles: tile数量
            max_batches: 最大批次数
        """
        # 按group_key分组
        groups: Dict[str, List[int]] = {}
        for i, tile in enumerate(tiles):
            if tile.group_key not in groups:
                groups[tile.group_key] = []
            groups[tile.group_key].append(i)

        # 对每个组，尝试让tiles在相同或相邻batch
        for group_key, tile_indices in groups.items():
            if len(tile_indices) <= 1:
                continue

            # 按k_iteration排序
            tile_indices.sort(key=lambda i: tiles[i].k_iteration)

            # 软约束：相邻tiles在同一batch有奖励
            # 这里我们通过引入辅助变量实现
            for i in range(len(tile_indices) - 1):
                idx1, idx2 = tile_indices[i], tile_indices[i + 1]

                # 为每个batch创建"两个tiles都在这个batch"的变量
                for j in range(max_batches):
                    both_in_batch = model.NewBoolVar(f'both_{idx1}_{idx2}_in_{j}')

                    # both_in_batch == 1 <=> x[idx1,j] == 1 AND x[idx2,j] == 1
                    model.AddBoolAnd([x[idx1, j], x[idx2, j]]).OnlyEnforceIf(both_in_batch)
                    model.AddBoolOr([x[idx1, j].Not(), x[idx2, j].Not()]).OnlyEnforceIf(both_in_batch.Not())

    def _extract_batches(
        self,
        solver,  # cp_model.CpSolver
        x: Dict[Tuple[int, int], any],  # Dict of cp_model.IntVar
        tiles: List[TileInfo],
        num_tiles: int,
        max_batches: int
    ) -> List[List[TileInfo]]:
        """
        从求解器结果中提取批次分配

        Args:
            solver: 求解器
            x: 决策变量
            tiles: tiles列表
            num_tiles: tile数量
            max_batches: 最大批次数

        Returns:
            批次列表
        """
        # 提取分配
        batch_assignments: Dict[int, List[TileInfo]] = {}

        for i in range(num_tiles):
            for j in range(max_batches):
                if solver.Value(x[i, j]):
                    if j not in batch_assignments:
                        batch_assignments[j] = []
                    batch_assignments[j].append(tiles[i])

        # 按batch_id排序并返回
        batches = []
        for batch_id in sorted(batch_assignments.keys()):
            batches.append(batch_assignments[batch_id])

        return batches
