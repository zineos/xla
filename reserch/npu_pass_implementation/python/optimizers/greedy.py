"""
贪心批调度算法

基础贪心算法：按顺序填充批次，每个批次填满16个tiles后开始新批次。
适用场景：单矩阵、快速编译
"""

import time
from typing import List
from ..core.batch_optimizer import BatchOptimizer, BatchScheduleResult
from ..core.tile_info import TileInfo
from ..core.hardware_config import HardwareConfig


class GreedyOptimizer(BatchOptimizer):
    """
    贪心批调度算法

    算法逻辑：
    1. 按tile_id顺序处理
    2. 当前批次未满则添加tile
    3. 当前批次满了则开始新批次
    4. 如果must_fill_batch=True，最后一个批次不足时用NOP填充
    """

    def __init__(self, config: HardwareConfig):
        super().__init__(config)

    def get_algorithm_name(self) -> str:
        return "greedy"

    def optimize(
        self,
        tiles: List[TileInfo],
        **kwargs
    ) -> BatchScheduleResult:
        """
        使用贪心算法调度tiles

        Args:
            tiles: 待调度的tiles
            **kwargs: 未使用

        Returns:
            BatchScheduleResult
        """
        start_time = time.time()

        # 验证输入
        self.validate_tiles(tiles)

        # 按tile_id排序（确保确定性）
        sorted_tiles = sorted(tiles, key=lambda t: t.tile_id)

        # 分配到批次
        batches = []
        current_batch = []

        for tile in sorted_tiles:
            current_batch.append(tile)

            # 批次满了
            if len(current_batch) == self.batch_size:
                batches.append(current_batch)
                current_batch = []

        # 处理最后一个未满的批次
        if current_batch:
            if self.must_fill_batch:
                # 必须填满：保留这个批次（会用NOP填充）
                batches.append(current_batch)
            else:
                # 可以不满：保留这个批次
                batches.append(current_batch)

        # 计算统计信息
        num_batches = len(batches)
        total_nops = self.count_nops(batches)
        utilization = self.calculate_utilization(batches)
        execution_time = time.time() - start_time

        return BatchScheduleResult(
            batches=batches,
            num_batches=num_batches,
            total_nops=total_nops,
            utilization=utilization,
            algorithm=self.get_algorithm_name(),
            execution_time=execution_time,
            metadata={
                "total_tiles": len(tiles),
                "batch_size": self.batch_size,
            }
        )
