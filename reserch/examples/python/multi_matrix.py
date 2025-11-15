#!/usr/bin/env python3
"""
多矩阵批组合优化示例

演示跨多个矩阵的tile批处理优化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "npu_pass_implementation"))

from python.core.tile_info import TileInfo
from python.core.hardware_config import HardwareConfig
from python.optimizers.improved_greedy import ImprovedGreedyOptimizer
from python.optimizers.cpsat_optimizer import CPSATOptimizer


def generate_tiles_for_matmul(matrix_id, M, K, N, tile_size=(16, 16, 16)):
    """为单个matmul生成tiles"""
    tiles = []
    tile_id_offset = matrix_id * 1000  # 不同矩阵用不同的ID范围

    tileM, tileN, tileK = tile_size

    for m in range(0, M, tileM):
        for n in range(0, N, tileN):
            for k in range(0, K, tileK):
                actual_m = min(tileM, M - m)
                actual_n = min(tileN, N - n)
                actual_k = min(tileK, K - k)

                tile = TileInfo(
                    tile_id=len(tiles) + tile_id_offset,
                    matrix_id=matrix_id,
                    m_offset=m,
                    n_offset=n,
                    k_iteration=k // tileK,
                    actual_m=actual_m,
                    actual_n=actual_n,
                    actual_k=actual_k,
                    a_slice=(m, m + actual_m, k, k + actual_k),
                    b_slice=(k, k + actual_k, n, n + actual_n),
                    c_slice=(m, m + actual_m, n, n + actual_n),
                    accumulate=(k > 0)
                )
                tiles.append(tile)

    return tiles


def create_multi_matrix_scenario():
    """创建多矩阵场景"""
    all_tiles = []

    # 矩阵0: 中等大小 100x200 × 200x100
    print("矩阵0: 100x200 × 200x100")
    tiles0 = generate_tiles_for_matmul(0, M=100, K=200, N=100)
    print(f"  生成 {len(tiles0)} tiles")
    all_tiles.extend(tiles0)

    # 矩阵1: 小矩阵 48x64 × 64x48
    print("矩阵1: 48x64 × 64x48")
    tiles1 = generate_tiles_for_matmul(1, M=48, K=64, N=48)
    print(f"  生成 {len(tiles1)} tiles")
    all_tiles.extend(tiles1)

    # 矩阵2: 不规则矩阵 35x50 × 50x40
    print("矩阵2: 35x50 × 50x40 (不规则)")
    tiles2 = generate_tiles_for_matmul(2, M=35, K=50, N=40)
    print(f"  生成 {len(tiles2)} tiles")
    all_tiles.extend(tiles2)

    return all_tiles, [len(tiles0), len(tiles1), len(tiles2)]


def analyze_batches(batches, tile_counts):
    """分析batch组成"""
    matrix_in_batches = [set() for _ in batches]

    for batch_id, batch in enumerate(batches):
        for tile in batch:
            matrix_in_batches[batch_id].add(tile.matrix_id)

    # 统计跨矩阵的batch
    cross_matrix_batches = sum(1 for matrices in matrix_in_batches if len(matrices) > 1)

    return matrix_in_batches, cross_matrix_batches


def main():
    """主函数"""
    print("=" * 70)
    print("多矩阵批组合优化示例")
    print("=" * 70)
    print()

    # 创建硬件配置
    config = HardwareConfig.get_default("npu")
    print(f"硬件: {config.name} (batch_size={config.compute.batch_size})")
    print()

    # 创建多矩阵场景
    print("创建多矩阵场景:")
    print("-" * 70)
    all_tiles, tile_counts = create_multi_matrix_scenario()
    print(f"\n总tile数: {len(all_tiles)}")
    print()

    # 使用ImprovedGreedy优化
    print("=" * 70)
    print("运行ImprovedGreedy优化 (支持跨矩阵)")
    print("=" * 70)

    optimizer = ImprovedGreedyOptimizer(config)
    result = optimizer.optimize(all_tiles)

    print(f"批次数: {result.num_batches}")
    print(f"NOP数: {result.total_nops}")
    print(f"利用率: {result.utilization:.2%}")
    print()

    # 分析batch组成
    matrix_in_batches, cross_matrix = analyze_batches(result.batches, tile_counts)

    print("Batch详细信息:")
    print("-" * 70)
    for i, (batch, matrices) in enumerate(zip(result.batches, matrix_in_batches)):
        cross_mark = "🔄" if len(matrices) > 1 else "  "
        print(f"{cross_mark} Batch {i}: {len(batch)} tiles, 来自矩阵 {sorted(matrices)}")

        # 显示每个矩阵的tile数
        for matrix_id in sorted(matrices):
            count = sum(1 for tile in batch if tile.matrix_id == matrix_id)
            print(f"     矩阵{matrix_id}: {count} tiles")

    print()
    print(f"跨矩阵批次: {cross_matrix}/{result.num_batches} ({cross_matrix/result.num_batches:.1%})")
    print()

    # 使用CP-SAT优化（如果tile数不太多）
    if len(all_tiles) <= 100:
        print("=" * 70)
        print("运行CP-SAT优化 (全局最优)")
        print("=" * 70)

        cpsat_optimizer = CPSATOptimizer(config)
        cpsat_result = cpsat_optimizer.optimize(all_tiles, time_limit=10.0)

        print(f"批次数: {cpsat_result.num_batches}")
        print(f"NOP数: {cpsat_result.total_nops}")
        print(f"利用率: {cpsat_result.utilization:.2%}")
        print(f"求解时间: {cpsat_result.execution_time:.2f}s")
        print()

        # 对比
        print("=" * 70)
        print("性能对比")
        print("=" * 70)
        batch_improvement = (result.num_batches - cpsat_result.num_batches) / result.num_batches * 100
        nop_improvement = (result.total_nops - cpsat_result.total_nops) / max(result.total_nops, 1) * 100

        print(f"CP-SAT vs ImprovedGreedy:")
        print(f"  批次数: {cpsat_result.num_batches} vs {result.num_batches} ({batch_improvement:+.1f}%)")
        print(f"  NOP数: {cpsat_result.total_nops} vs {result.total_nops} ({nop_improvement:+.1f}%)")
        print(f"  利用率: {cpsat_result.utilization:.2%} vs {result.utilization:.2%}")

    print("\n" + "=" * 70)
    print("✅ 多矩阵优化完成!")
    print("=" * 70)
    print("\n💡 关键观察:")
    print("  - 跨矩阵批处理提高了硬件利用率")
    print("  - K-dimension分组确保累加正确性")
    print("  - CP-SAT能找到更优的全局解（但耗时更长）")

    return 0


if __name__ == "__main__":
    sys.exit(main())
