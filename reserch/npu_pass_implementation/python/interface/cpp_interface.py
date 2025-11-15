#!/usr/bin/env python3
"""
C++ Interface for Batch Composition Optimizer

This module provides a clean, stable interface for C++ code to call Python
batch composition optimizers. It handles data conversion and error handling.

Usage from C++ (via pybind11):
    import python.interface.cpp_interface as interface
    result = interface.optimize_tiles(tiles_data, config)
"""

import sys
import json
from typing import List, Dict, Any, Optional
from dataclasses import asdict

# Import core modules
try:
    from ..core.tile_info import TileInfo
    from ..core.hardware_config import HardwareConfig, HardwareType
    from ..core.batch_optimizer import UnifiedOptimizer, BatchScheduleResult
except ImportError:
    # Fallback for direct execution
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from core.tile_info import TileInfo
    from core.hardware_config import HardwareConfig, HardwareType
    from core.batch_optimizer import UnifiedOptimizer, BatchScheduleResult


class CPPInterface:
    """
    C++ Interface for Batch Composition Optimizer

    This class provides a stable API for C++ code to interact with Python
    optimizers. It handles all data conversion and error handling.
    """

    def __init__(self, hardware_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the C++ interface

        Args:
            hardware_config: Hardware configuration dictionary
                {
                    "hardware_type": "npu" | "gpu" | "tpu",
                    "batch_size": int,
                    "config_path": str (optional)
                }
        """
        if hardware_config is None:
            hardware_config = {"hardware_type": "npu"}

        self.hardware_type = hardware_config.get("hardware_type", "npu")
        config_path = hardware_config.get("config_path")

        # Load hardware configuration
        if config_path:
            self.config = HardwareConfig.from_yaml(config_path)
        else:
            self.config = HardwareConfig.get_default(self.hardware_type)

        # Create unified optimizer
        self.optimizer = UnifiedOptimizer(self.config)

    def optimize(
        self,
        tiles_data: List[Dict[str, Any]],
        algorithm: str = "auto"
    ) -> Dict[str, Any]:
        """
        Optimize tile-to-batch assignment

        Args:
            tiles_data: List of tile dictionaries from C++
                Each tile: {
                    "tile_id": int,
                    "matrix_id": int,
                    "m_offset": int,
                    "n_offset": int,
                    "k_iteration": int,
                    "actual_m": int,
                    "actual_n": int,
                    "actual_k": int,
                    "a_slice": [m_start, m_end, k_start, k_end],
                    "b_slice": [k_start, k_end, n_start, n_end],
                    "c_slice": [m_start, m_end, n_start, n_end],
                    "accumulate": bool
                }

            algorithm: Optimization algorithm
                "auto", "greedy", "improved_greedy", "cpsat"

        Returns:
            Result dictionary:
                {
                    "batches": [
                        [tile_id, tile_id, ...],  # batch 0
                        [tile_id, tile_id, ...],  # batch 1
                        ...
                    ],
                    "num_batches": int,
                    "total_nops": int,
                    "utilization": float,
                    "algorithm": str,
                    "execution_time": float,
                    "metadata": {...}
                }
        """
        try:
            # Convert C++ tile data to Python TileInfo objects
            tiles = self._tiles_from_cpp(tiles_data)

            # Run optimization
            result = self.optimizer.optimize(tiles, algorithm=algorithm)

            # Convert result to C++ format
            return self._result_to_cpp(result)

        except Exception as e:
            # Return error result
            return {
                "error": str(e),
                "batches": [],
                "num_batches": 0,
                "total_nops": 0,
                "utilization": 0.0,
                "algorithm": algorithm,
                "execution_time": 0.0
            }

    def _tiles_from_cpp(self, tiles_data: List[Dict[str, Any]]) -> List[TileInfo]:
        """
        Convert C++ tile data to Python TileInfo objects

        Args:
            tiles_data: List of tile dictionaries from C++

        Returns:
            List of TileInfo objects
        """
        tiles = []

        for data in tiles_data:
            tile = TileInfo(
                tile_id=data["tile_id"],
                matrix_id=data["matrix_id"],
                m_offset=data["m_offset"],
                n_offset=data["n_offset"],
                k_iteration=data["k_iteration"],
                actual_m=data["actual_m"],
                actual_n=data["actual_n"],
                actual_k=data["actual_k"],
                a_slice=tuple(data["a_slice"]),
                b_slice=tuple(data["b_slice"]),
                c_slice=tuple(data["c_slice"]),
                accumulate=data["accumulate"]
            )
            tiles.append(tile)

        return tiles

    def _result_to_cpp(self, result: BatchScheduleResult) -> Dict[str, Any]:
        """
        Convert Python BatchScheduleResult to C++ format

        Args:
            result: Python batch schedule result

        Returns:
            Dictionary for C++ consumption
        """
        # Convert batches: List[List[TileInfo]] -> List[List[int]]
        batches_ids = []
        for batch in result.batches:
            batch_ids = [tile.tile_id for tile in batch]
            batches_ids.append(batch_ids)

        return {
            "batches": batches_ids,
            "num_batches": result.num_batches,
            "total_nops": result.total_nops,
            "utilization": result.utilization,
            "algorithm": result.algorithm,
            "execution_time": result.execution_time,
            "metadata": result.metadata
        }


# Global interface instance (singleton pattern)
_global_interface: Optional[CPPInterface] = None


def initialize_interface(hardware_config: Optional[Dict[str, Any]] = None):
    """
    Initialize the global C++ interface

    Args:
        hardware_config: Hardware configuration dictionary
    """
    global _global_interface
    _global_interface = CPPInterface(hardware_config)


def optimize_tiles(
    tiles_data: List[Dict[str, Any]],
    algorithm: str = "auto",
    hardware_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Optimize tiles (simplified interface for C++)

    This is the main entry point for C++ code.

    Args:
        tiles_data: List of tile dictionaries
        algorithm: Optimization algorithm
        hardware_config: Hardware configuration (optional)

    Returns:
        Result dictionary
    """
    global _global_interface

    # Initialize interface if not already done
    if _global_interface is None:
        initialize_interface(hardware_config)

    return _global_interface.optimize(tiles_data, algorithm)


def optimize_tiles_from_json(json_str: str) -> str:
    """
    Optimize tiles from JSON string (alternative interface)

    This is useful for C++ code that prefers JSON serialization.

    Args:
        json_str: JSON string containing:
            {
                "tiles": [...],
                "algorithm": "auto",
                "hardware_config": {...}
            }

    Returns:
        JSON string containing result
    """
    try:
        data = json.loads(json_str)

        tiles_data = data.get("tiles", [])
        algorithm = data.get("algorithm", "auto")
        hardware_config = data.get("hardware_config")

        result = optimize_tiles(tiles_data, algorithm, hardware_config)

        return json.dumps(result, indent=2)

    except Exception as e:
        error_result = {
            "error": str(e),
            "batches": [],
            "num_batches": 0
        }
        return json.dumps(error_result)


# ============================================================================
# Testing and Debugging
# ============================================================================

def test_interface():
    """Test the C++ interface with sample data"""

    print("=" * 60)
    print("Testing C++ Interface")
    print("=" * 60)

    # Sample tile data (simulating C++ input)
    tiles_data = []
    for i in range(20):
        tile = {
            "tile_id": i,
            "matrix_id": 0,
            "m_offset": 0,
            "n_offset": 0,
            "k_iteration": i,
            "actual_m": 16,
            "actual_n": 16,
            "actual_k": 16,
            "a_slice": [0, 16, i * 16, (i + 1) * 16],
            "b_slice": [i * 16, (i + 1) * 16, 0, 16],
            "c_slice": [0, 16, 0, 16],
            "accumulate": i > 0
        }
        tiles_data.append(tile)

    print(f"\n[1] Created {len(tiles_data)} tiles")

    # Test with different algorithms
    for algorithm in ["greedy", "improved_greedy"]:
        print(f"\n[2] Testing {algorithm} algorithm...")

        result = optimize_tiles(
            tiles_data,
            algorithm=algorithm,
            hardware_config={"hardware_type": "npu"}
        )

        print(f"  ✓ Algorithm: {result['algorithm']}")
        print(f"  ✓ Batches: {result['num_batches']}")
        print(f"  ✓ NOPs: {result['total_nops']}")
        print(f"  ✓ Utilization: {result['utilization']:.2%}")
        print(f"  ✓ Time: {result['execution_time']:.4f}s")

    # Test JSON interface
    print(f"\n[3] Testing JSON interface...")

    json_input = json.dumps({
        "tiles": tiles_data,
        "algorithm": "greedy",
        "hardware_config": {"hardware_type": "npu"}
    })

    json_output = optimize_tiles_from_json(json_input)
    result = json.loads(json_output)

    print(f"  ✓ JSON input size: {len(json_input)} bytes")
    print(f"  ✓ JSON output size: {len(json_output)} bytes")
    print(f"  ✓ Result batches: {result['num_batches']}")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    # Run tests when executed directly
    test_interface()
