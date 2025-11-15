#!/usr/bin/env python3
"""
测试新的批组合优化框架结构

验证所有模块可以正确导入和使用。
"""

import sys
from pathlib import Path

# 添加父目录到路径（使python成为一个包）
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 60)
print("测试批组合优化框架 - 新结构验证")
print("=" * 60)

# 测试1: 导入核心模块
print("\n[1] 测试核心模块导入...")
try:
    from python.core.tile_info import TileInfo
    from python.core.hardware_config import HardwareConfig, HardwareType
    from python.core.batch_optimizer import UnifiedOptimizer, BatchScheduleResult
    print("✓ 核心模块导入成功")
except Exception as e:
    print(f"✗ 核心模块导入失败: {e}")
    sys.exit(1)

# 测试2: 导入优化器模块
print("\n[2] 测试优化器模块导入...")
try:
    from python.optimizers.greedy import GreedyOptimizer
    from python.optimizers.improved_greedy import ImprovedGreedyOptimizer
    print("✓ 优化器模块导入成功")
except Exception as e:
    print(f"✗ 优化器模块导入失败: {e}")
    sys.exit(1)

# 测试3: 导入CP-SAT优化器（可选）
print("\n[3] 测试CP-SAT优化器导入...")
try:
    from python.optimizers.cpsat_optimizer import CPSATOptimizer
    print("✓ CP-SAT优化器导入成功")
    HAS_CPSAT = True
except ImportError:
    print("⚠ CP-SAT优化器不可用（需要安装ortools）")
    HAS_CPSAT = False

# 测试4: 创建硬件配置
print("\n[4] 测试硬件配置...")
try:
    # NPU配置
    npu_config = HardwareConfig.get_default(HardwareType.NPU)
    print(f"✓ NPU配置: batch_size={npu_config.compute.batch_size}")

    # GPU配置
    gpu_config = HardwareConfig.get_default(HardwareType.GPU)
    print(f"✓ GPU配置: batch_size={gpu_config.compute.batch_size}")

    # TPU配置
    tpu_config = HardwareConfig.get_default(HardwareType.TPU)
    print(f"✓ TPU配置: batch_size={tpu_config.compute.batch_size}")
except Exception as e:
    print(f"✗ 硬件配置创建失败: {e}")
    sys.exit(1)

# 测试5: 创建tiles
print("\n[5] 测试TileInfo创建...")
try:
    tiles = []
    for i in range(20):
        tile = TileInfo(
            tile_id=i,
            matrix_id=0,
            m_offset=0,
            n_offset=0,
            k_iteration=i,
            actual_m=16,
            actual_n=16,
            actual_k=16,
            a_slice=(0, 16, i * 16, (i + 1) * 16),
            b_slice=(i * 16, (i + 1) * 16, 0, 16),
            c_slice=(0, 16, 0, 16),
            accumulate=(i > 0),
        )
        tiles.append(tile)
    print(f"✓ 创建了 {len(tiles)} 个tiles")
except Exception as e:
    print(f"✗ TileInfo创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试6: 贪心优化器
print("\n[6] 测试贪心优化器...")
try:
    optimizer = GreedyOptimizer(npu_config)
    result = optimizer.optimize(tiles)
    print(f"✓ 贪心算法:")
    print(f"  - Batches: {result.num_batches}")
    print(f"  - NOPs: {result.total_nops}")
    print(f"  - 利用率: {result.utilization:.2%}")
    print(f"  - 时间: {result.execution_time:.4f}s")
except Exception as e:
    print(f"✗ 贪心优化器失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试7: 改进贪心优化器
print("\n[7] 测试改进贪心优化器...")
try:
    optimizer = ImprovedGreedyOptimizer(npu_config)
    result = optimizer.optimize(tiles)
    print(f"✓ 改进贪心算法:")
    print(f"  - Batches: {result.num_batches}")
    print(f"  - NOPs: {result.total_nops}")
    print(f"  - 利用率: {result.utilization:.2%}")
    print(f"  - 时间: {result.execution_time:.4f}s")
except Exception as e:
    print(f"✗ 改进贪心优化器失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试8: CP-SAT优化器（如果可用）
if HAS_CPSAT:
    print("\n[8] 测试CP-SAT优化器...")
    try:
        optimizer = CPSATOptimizer(npu_config, time_limit=2.0)
        result = optimizer.optimize(tiles)
        print(f"✓ CP-SAT算法:")
        print(f"  - Batches: {result.num_batches}")
        print(f"  - NOPs: {result.total_nops}")
        print(f"  - 利用率: {result.utilization:.2%}")
        print(f"  - 时间: {result.execution_time:.4f}s")
    except Exception as e:
        print(f"✗ CP-SAT优化器失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("\n[8] 跳过CP-SAT优化器测试（未安装）")

# 测试9: 统一优化器
print("\n[9] 测试统一优化器...")
try:
    unified = UnifiedOptimizer(npu_config)

    # 自动选择
    result_auto = unified.optimize(tiles, algorithm="auto")
    print(f"✓ 自动选择: {result_auto.algorithm}")

    # 强制使用贪心
    result_greedy = unified.optimize(tiles, algorithm="greedy")
    print(f"✓ 强制贪心: {result_greedy.algorithm}")

    # 强制使用改进贪心
    result_improved = unified.optimize(tiles, algorithm="improved_greedy")
    print(f"✓ 强制改进贪心: {result_improved.algorithm}")
except Exception as e:
    print(f"✗ 统一优化器失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试10: 多矩阵场景
print("\n[10] 测试多矩阵场景...")
try:
    multi_tiles = []
    tile_id = 0

    # 创建3个矩阵，每个矩阵不同数量的tiles
    for matrix_id in range(3):
        num_tiles_per_matrix = 10 + matrix_id * 5
        for i in range(num_tiles_per_matrix):
            tile = TileInfo(
                tile_id=tile_id,
                matrix_id=matrix_id,
                m_offset=0,
                n_offset=0,
                k_iteration=i,
                actual_m=16,
                actual_n=16,
                actual_k=16,
                a_slice=(0, 16, i * 16, (i + 1) * 16),
                b_slice=(i * 16, (i + 1) * 16, 0, 16),
                c_slice=(0, 16, 0, 16),
                accumulate=(i > 0),
            )
            multi_tiles.append(tile)
            tile_id += 1

    print(f"  创建了 {len(multi_tiles)} 个tiles，来自 3 个矩阵")

    # 使用改进贪心
    optimizer = ImprovedGreedyOptimizer(npu_config)
    result = optimizer.optimize(multi_tiles)
    print(f"✓ 改进贪心（多矩阵）:")
    print(f"  - Batches: {result.num_batches}")
    print(f"  - NOPs: {result.total_nops}")
    print(f"  - 利用率: {result.utilization:.2%}")
    print(f"  - 矩阵数: {result.metadata['num_matrices']}")
    print(f"  - 分组数: {result.metadata['num_groups']}")
except Exception as e:
    print(f"✗ 多矩阵场景失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✓ 所有测试通过！新框架结构验证成功")
print("=" * 60)
