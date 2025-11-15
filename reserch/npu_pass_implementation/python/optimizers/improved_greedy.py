"""
改进贪心批调度算法

改进的贪心算法：考虑K维规约分组和跨矩阵填充优化。
适用场景：多矩阵、中小规模（<1000 tiles）
"""

import time
from typing import List, Dict, Set
from ..core.batch_optimizer import BatchOptimizer, BatchScheduleResult
from ..core.tile_info import TileInfo
from ..core.hardware_config import HardwareConfig


class ImprovedGreedyOptimizer(BatchOptimizer):
    """
    改进贪心批调度算法

    算法逻辑：
    1. 按group_key分组tiles（同一输出位置的K维tiles在同一组）
    2. 优先处理大组（减少碎片）
    3. 组内按k_iteration排序（保证规约顺序）
    4. 当前组的批次未满时，尝试从其他组填充
    5. 这样可以减少NOPs，提高利用率
    """

    def __init__(self, config: HardwareConfig):
        super().__init__(config)

    def get_algorithm_name(self) -> str:
        return "improved_greedy"

    def optimize(
        self,
        tiles: List[TileInfo],
        **kwargs
    ) -> BatchScheduleResult:
        """
        使用改进贪心算法调度tiles

        Args:
            tiles: 待调度的tiles
            **kwargs: 未使用

        Returns:
            BatchScheduleResult
        """
        start_time = time.time()

        # 验证输入
        self.validate_tiles(tiles)

        # Step 1: 按group_key分组
        tile_groups = self._group_tiles_by_output_location(tiles)

        # Step 2: 按组大小排序（大组优先）
        sorted_groups = sorted(
            tile_groups.items(),
            key=lambda x: len(x[1]),
            reverse=True
        )

        # Step 3: 依次处理每个组
        batches = []
        processed: Set[int] = set()
        current_batch: List[TileInfo] = []

        for group_key, group_tiles in sorted_groups:
            # 组内按k_iteration排序
            group_tiles.sort(key=lambda t: t.k_iteration)

            for tile in group_tiles:
                if tile.tile_id in processed:
                    continue

                current_batch.append(tile)
                processed.add(tile.tile_id)

                # 批次满了
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []

            # Step 4: 当前组处理完毕，如果批次未满，尝试从其他组填充
            if current_batch and len(current_batch) < self.batch_size:
                self._fill_batch_from_other_groups(
                    current_batch,
                    sorted_groups,
                    group_key,
                    processed
                )

                # 检查是否填满
                if len(current_batch) == self.batch_size:
                    batches.append(current_batch)
                    current_batch = []

        # 处理最后一个未满的批次
        if current_batch:
            batches.append(current_batch)

        # 计算统计信息
        num_batches = len(batches)
        total_nops = self.count_nops(batches)
        utilization = self.calculate_utilization(batches)
        execution_time = time.time() - start_time

        # 额外的统计信息
        num_matrices = len(set(t.matrix_id for t in tiles))
        num_groups = len(tile_groups)

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
            }
        )

    def _group_tiles_by_output_location(
        self,
        tiles: List[TileInfo]
    ) -> Dict[str, List[TileInfo]]:
        """
        按输出位置分组tiles

        同一个输出位置的不同K维迭代tiles应该在同一组。

        Returns:
            {group_key: [tiles]}字典
        """
        groups: Dict[str, List[TileInfo]] = {}

        for tile in tiles:
            # group_key已经由TileInfo生成
            if tile.group_key not in groups:
                groups[tile.group_key] = []
            groups[tile.group_key].append(tile)

        return groups

    def _fill_batch_from_other_groups(
        self,
        current_batch: List[TileInfo],
        sorted_groups,
        current_group_key: str,
        processed: Set[int]
    ) -> None:
        """
        从其他组填充当前批次

        Args:
            current_batch: 当前未满的批次
            sorted_groups: 排序后的分组列表
            current_group_key: 当前组的key
            processed: 已处理的tile_id集合
        """
        for other_group_key, other_tiles in sorted_groups:
            # 跳过当前组
            if other_group_key == current_group_key:
                continue

            # 尝试从其他组取tiles
            for tile in other_tiles:
                if tile.tile_id in processed:
                    continue

                current_batch.append(tile)
                processed.add(tile.tile_id)

                # 批次满了就停止
                if len(current_batch) == self.batch_size:
                    return

            # 批次满了就停止
            if len(current_batch) == self.batch_size:
                return
