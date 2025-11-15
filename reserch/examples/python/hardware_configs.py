#!/usr/bin/env python3
"""
硬件配置使用示例

演示如何使用不同硬件配置进行优化
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "npu_pass_implementation"))

from python.core.tile_info import TileInfo
from python.core.hardware_config import HardwareConfig, HardwareType
from python.optimizers.improved_greedy import ImprovedGreedyOptimizer


def create_sample_tiles(num_tiles=50):
    """创建示例tiles"""
    tiles = []
    for i in range(num_tiles):
        tile = TileInfo(
            tile_id=i,
            matrix_id=0,
            m_offset=(i % 7) * 16,
            n_offset=(i % 5) * 16,
            k_iteration=i // 35,
            actual_m=16,
            actual_n=16,
            actual_k=16,
            a_slice=(0, 16, 0, 16),
            b_slice=(0, 16, 0, 16),
            c_slice=(0, 16, 0, 16),
            accumulate=(i >= 35)
        )
        tiles.append(tile)
    return tiles


def test_hardware_config(hw_type: str, tiles):
    """测试特定硬件配置"""
    print(f"\n{'=' * 70}")
    print(f"{hw_type.upper()} 硬件配置")
    print('=' * 70)

    # 获取硬件配置
    config = HardwareConfig.get_default(hw_type)

    # 显示配置信息
    print(f"\n硬件参数:")
    print(f"  名称: {config.name}")
    print(f"  类型: {config.hardware_type.value}")
    print(f"  Batch大小: {config.compute.batch_size}")
    print(f"  必须填满: {config.optimization.must_fill_batch}")
    print(f"  支持动态batch: {config.optimization.allow_dynamic_batch}")
    print(f"  最大NOPs: {config.optimization.max_nops_per_batch}")

    # 运行优化
    print(f"\n运行优化...")
    optimizer = ImprovedGreedyOptimizer(config)
    result = optimizer.optimize(tiles)

    # 显示结果
    print(f"\n优化结果:")
    print(f"  批次数: {result.num_batches}")
    print(f"  总NOP数: {result.total_nops}")
    print(f"  平均利用率: {result.utilization:.2%}")
    print(f"  执行时间: {result.execution_time:.4f}s")

    # 显示batch详情
    print(f"\nBatch详情:")
    for i, batch in enumerate(result.batches[:5]):  # 只显示前5个
        nops = config.compute.batch_size - len(batch)
        util = len(batch) / config.compute.batch_size
        print(f"  Batch {i}: {len(batch)} tiles, {nops} NOPs, 利用率 {util:.1%}")

    if len(result.batches) > 5:
        print(f"  ... 还有 {len(result.batches) - 5} 个batches")

    return result


def compare_hardware():
    """比较不同硬件配置的性能"""
    print("\n" + "=" * 70)
    print("硬件配置对比")
    print("=" * 70)

    # 创建测试tiles
    tiles = create_sample_tiles(num_tiles=50)
    print(f"\n测试数据: {len(tiles)} tiles")

    # 测试三种硬件
    results = {}
    for hw in ["npu", "gpu", "tpu"]:
        results[hw] = test_hardware_config(hw, tiles)

    # 对比结果
    print("\n" + "=" * 70)
    print("性能对比总结")
    print("=" * 70)

    print(f"\n{'硬件':<10} {'Batch数':<10} {'NOP数':<10} {'利用率':<12} {'Batch大小'}")
    print("-" * 70)

    for hw in ["npu", "gpu", "tpu"]:
        result = results[hw]
        config = HardwareConfig.get_default(hw)
        print(f"{hw.upper():<10} {result.num_batches:<10} {result.total_nops:<10} "
              f"{result.utilization:<12.2%} {config.compute.batch_size}")

    print()


def test_custom_config():
    """测试自定义配置"""
    print("\n" + "=" * 70)
    print("自定义硬件配置")
    print("=" * 70)

    from python.core.hardware_config import ComputeConfig, MemoryConfig, OptimizationConfig

    # 创建自定义配置
    custom_config = HardwareConfig(
        name="Custom Accelerator",
        hardware_type=HardwareType.NPU,
        compute=ComputeConfig(
            batch_size=24,  # 自定义batch大小
            tile_size=(16, 16, 16),
            vector_lanes=24,
        ),
        memory=MemoryConfig(
            shared_memory_kb=64,
            register_file_kb=16,
        ),
        optimization=OptimizationConfig(
            must_fill_batch=False,  # 不强制填满
            allow_dynamic_batch=True,
            max_nops_per_batch=8,  # 最多8个NOPs
        )
    )

    print(f"\n自定义配置:")
    print(f"  Batch大小: {custom_config.compute.batch_size}")
    print(f"  必须填满: {custom_config.optimization.must_fill_batch}")
    print(f"  最大NOPs: {custom_config.optimization.max_nops_per_batch}")

    # 测试
    tiles = create_sample_tiles(num_tiles=50)
    optimizer = ImprovedGreedyOptimizer(custom_config)
    result = optimizer.optimize(tiles)

    print(f"\n优化结果:")
    print(f"  批次数: {result.num_batches}")
    print(f"  NOP数: {result.total_nops}")
    print(f"  利用率: {result.utilization:.2%}")


def main():
    """主函数"""
    print("=" * 70)
    print("BatchComp硬件配置示例")
    print("=" * 70)

    try:
        # 对比不同硬件
        compare_hardware()

        # 自定义配置
        test_custom_config()

        print("\n" + "=" * 70)
        print("✅ 硬件配置示例运行成功!")
        print("=" * 70)

        print("\n💡 关键观察:")
        print("  - NPU需要填满batch (must_fill=true)")
        print("  - GPU/TPU更灵活 (must_fill=false)")
        print("  - 更大的batch size可能导致更多NOPs")
        print("  - 可以通过YAML文件自定义硬件配置")

        print("\n📝 下一步:")
        print("  - 查看 npu_pass_implementation/configs/*.yaml 了解配置文件格式")
        print("  - 修改 max_nops_per_batch 观察优化行为变化")
        print("  - 尝试不同的 tile_size 组合")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
