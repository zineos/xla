#!/usr/bin/env python3
"""
BatchComp基础优化示例

演示如何使用Python优化器进行tile批处理优化
"""

import sys
from pathlib import Path

# 添加BatchComp Python包到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "npu_pass_implementation"))

from python.core.tile_info import TileInfo
from python.core.hardware_config import HardwareConfig
from python.optimizers.greedy import GreedyOptimizer
from python.optimizers.improved_greedy import ImprovedGreedyOptimizer
from python.optimizers.cpsat_optimizer import CPSATOptimizer


def create_sample_tiles(num_tiles=20, matrix_id=0):
    """创建示例tiles"""
    tiles = []
    tile_id = 0

    # 模拟100x100矩阵，tile大小16x16
    for m in range(0, 100, 16):
        for n in range(0, 100, 16):
            for k in range(0, 100, 16):
                tile = TileInfo(
                    tile_id=tile_id,
                    matrix_id=matrix_id,
                    m_offset=m,
                    n_offset=n,
                    k_iteration=k // 16,
                    actual_m=min(16, 100 - m),
                    actual_n=min(16, 100 - n),
                    actual_k=min(16, 100 - k),
                    a_slice=(m, m + 16, k, k + 16),
                    b_slice=(k, k + 16, n, n + 16),
                    c_slice=(m, m + 16, n, n + 16),
                    accumulate=(k > 0)
                )
                tiles.append(tile)
                tile_id += 1

                if tile_id >= num_tiles:
                    return tiles

    return tiles


def run_optimization_comparison():
    """运行不同算法的性能对比"""

    print("=" * 60)
    print("BatchComp基础优化示例")
    print("=" * 60)
    print()

    # 创建NPU硬件配置
    config = HardwareConfig.get_default("npu")
    print(f"硬件配置: {config.name}")
    print(f"  Batch大小: {config.compute.batch_size}")
    print(f"  必须填满: {config.optimization.must_fill_batch}")
    print()

    # 创建示例tiles
    tiles = create_sample_tiles(num_tiles=20)
    print(f"生成 {len(tiles)} 个tiles")
    print(f"  来自矩阵 #{tiles[0].matrix_id}")
    print(f"  Tile大小: {tiles[0].actual_m}x{tiles[0].actual_n}x{tiles[0].actual_k}")
    print()

    # 运行三种优化算法
    algorithms = [
        ("Greedy", GreedyOptimizer(config)),
        ("ImprovedGreedy", ImprovedGreedyOptimizer(config)),
        ("CP-SAT", CPSATOptimizer(config)),
    ]

    results = []

    for name, optimizer in algorithms:
        print(f"运行 {name} 算法...")
        result = optimizer.optimize(tiles)
        results.append((name, result))

        print(f"  批次数: {result.num_batches}")
        print(f"  NOP数: {result.total_nops}")
        print(f"  利用率: {result.utilization:.2%}")
        print(f"  执行时间: {result.execution_time:.4f}s")
        print()

    # 性能对比
    print("=" * 60)
    print("性能对比")
    print("=" * 60)

    baseline = results[0][1]  # Greedy作为基准

    for name, result in results:
        batch_improvement = (baseline.num_batches - result.num_batches) / baseline.num_batches * 100
        nop_improvement = (baseline.total_nops - result.total_nops) / max(baseline.total_nops, 1) * 100
        util_improvement = (result.utilization - baseline.utilization) * 100

        print(f"\n{name}:")
        print(f"  批次数: {result.num_batches} ({batch_improvement:+.1f}%)")
        print(f"  NOP数: {result.total_nops} ({nop_improvement:+.1f}%)")
        print(f"  利用率: {result.utilization:.2%} ({util_improvement:+.1f}%)")

        # 显示每个batch的详细信息
        if len(result.batches) <= 5:  # 只显示较小的结果
            print(f"  批次详情:")
            for i, batch in enumerate(result.batches):
                print(f"    Batch {i}: {len(batch)} tiles")


def main():
    """主函数"""
    try:
        run_optimization_comparison()

        print("\n" + "=" * 60)
        print("✅ 示例运行成功!")
        print("=" * 60)
        print("\n💡 提示:")
        print("  - 修改 num_tiles 参数尝试不同规模")
        print("  - 查看 hardware_configs.py 了解不同硬件配置")
        print("  - 查看 multi_matrix.py 了解多矩阵优化")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
