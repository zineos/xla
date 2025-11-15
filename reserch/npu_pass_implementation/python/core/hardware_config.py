"""
Hardware Configuration System

Provides a flexible configuration system for different hardware accelerators.
Supports NPU, GPU, TPU with their specific constraints and optimization parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple
from enum import Enum
import yaml
import json
from pathlib import Path


class HardwareType(Enum):
    """Supported hardware types"""
    NPU = "npu"
    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"  # For baseline comparison
    CUSTOM = "custom"


@dataclass
class MemoryConfig:
    """Memory hierarchy configuration"""
    on_chip_size: int  # Bytes
    l2_cache_size: Optional[int] = None
    ddr_bandwidth: float = 100.0  # GB/s
    dma_channels: int = 1
    supports_double_buffering: bool = False


@dataclass
class ComputeConfig:
    """Compute capability configuration"""
    batch_size: int  # Number of tiles per batch
    tile_size: Tuple[int, int, int]  # (M, N, K) dimensions
    must_fill_batch: bool  # Whether batch must be completely filled
    supports_variable_tiles: bool = False
    supports_mixed_precision: bool = False
    supported_dtypes: list = field(default_factory=lambda: ["float32"])
    compute_throughput: float = 1.0  # TFLOPS


@dataclass
class OptimizationConfig:
    """Optimization strategy configuration"""
    default_algorithm: str = "auto"
    cp_sat_time_limit: float = 10.0
    cp_sat_enabled_threshold: int = 50  # Use CP-SAT for <= this many tiles
    greedy_threshold: int = 1000  # Use greedy for > this many tiles
    enable_multi_matmul_fusion: bool = True
    enable_k_reduction_grouping: bool = True


@dataclass
class HardwareConfig:
    """Complete hardware configuration"""
    name: str
    hardware_type: HardwareType
    compute: ComputeConfig
    memory: MemoryConfig
    optimization: OptimizationConfig
    vendor_extensions: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'HardwareConfig':
        """Load configuration from YAML file"""
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def from_json(cls, json_path: str) -> 'HardwareConfig':
        """Load configuration from JSON file"""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {json_path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> 'HardwareConfig':
        """Create HardwareConfig from dictionary"""
        # Parse hardware type
        hw_type = HardwareType(data['hardware']['type'].lower())

        # Parse compute config
        compute_data = data.get('compute', {})
        compute = ComputeConfig(
            batch_size=compute_data.get('batch_size', 16),
            tile_size=tuple(compute_data.get('tile_size', [16, 16, 16])),
            must_fill_batch=compute_data.get('must_fill_batch', False),
            supports_variable_tiles=compute_data.get('supports_variable_tiles', False),
            supports_mixed_precision=compute_data.get('supports_mixed_precision', False),
            supported_dtypes=compute_data.get('supported_dtypes', ['float32']),
            compute_throughput=compute_data.get('compute_throughput', 1.0)
        )

        # Parse memory config
        memory_data = data.get('memory', {})
        memory = MemoryConfig(
            on_chip_size=memory_data.get('on_chip_buffer', 16384),
            l2_cache_size=memory_data.get('l2_cache'),
            ddr_bandwidth=memory_data.get('ddr_bandwidth', 100.0),
            dma_channels=memory_data.get('dma_channels', 1),
            supports_double_buffering=memory_data.get('double_buffer', False)
        )

        # Parse optimization config
        opt_data = data.get('optimization', {})
        optimization = OptimizationConfig(
            default_algorithm=opt_data.get('default_algorithm', 'auto'),
            cp_sat_time_limit=opt_data.get('cp_sat_time_limit', 10.0),
            cp_sat_enabled_threshold=opt_data.get('cp_sat_enabled_threshold', 50),
            greedy_threshold=opt_data.get('greedy_threshold', 1000),
            enable_multi_matmul_fusion=opt_data.get('enable_multi_matmul_fusion', True),
            enable_k_reduction_grouping=opt_data.get('enable_k_reduction_grouping', True)
        )

        return cls(
            name=data['hardware'].get('name', 'unknown'),
            hardware_type=hw_type,
            compute=compute,
            memory=memory,
            optimization=optimization,
            vendor_extensions=data.get('vendor_extensions', {})
        )

    @classmethod
    def get_default(cls, hardware) -> 'HardwareConfig':
        """
        Get default configuration for a hardware type

        Args:
            hardware: HardwareType enum or string ("npu", "gpu", "tpu")
        """
        # Convert HardwareType to string if needed
        if isinstance(hardware, HardwareType):
            hardware = hardware.value
        else:
            hardware = hardware.lower()

        if hardware == "npu":
            return cls.default_npu()
        elif hardware == "gpu":
            return cls.default_gpu()
        elif hardware == "tpu":
            return cls.default_tpu()
        else:
            raise ValueError(f"Unknown hardware type: {hardware}")

    @classmethod
    def default_npu(cls) -> 'HardwareConfig':
        """Default NPU configuration"""
        return cls(
            name="NPU-16",
            hardware_type=HardwareType.NPU,
            compute=ComputeConfig(
                batch_size=16,
                tile_size=(16, 16, 16),
                must_fill_batch=True,
                supports_variable_tiles=True,
                compute_throughput=0.5  # 0.5 TFLOPS
            ),
            memory=MemoryConfig(
                on_chip_size=16384,  # 16KB
                ddr_bandwidth=50.0,
                dma_channels=2,
                supports_double_buffering=True
            ),
            optimization=OptimizationConfig(
                default_algorithm="auto",
                cp_sat_time_limit=10.0,
                cp_sat_enabled_threshold=100,
                greedy_threshold=1000,
                enable_multi_matmul_fusion=True,
                enable_k_reduction_grouping=True
            )
        )

    @classmethod
    def default_gpu(cls) -> 'HardwareConfig':
        """Default GPU (Tensor Core) configuration"""
        return cls(
            name="GPU-TensorCore",
            hardware_type=HardwareType.GPU,
            compute=ComputeConfig(
                batch_size=32,  # Warp size
                tile_size=(16, 16, 16),  # Tensor Core tile size
                must_fill_batch=False,  # Can use partial warps
                supports_variable_tiles=False,
                supports_mixed_precision=True,
                supported_dtypes=["float16", "bfloat16", "float32", "int8"],
                compute_throughput=10.0  # 10 TFLOPS
            ),
            memory=MemoryConfig(
                on_chip_size=163840,  # 160KB shared memory per SM
                l2_cache_size=41943040,  # 40MB L2
                ddr_bandwidth=900.0,  # HBM bandwidth
                dma_channels=1,
                supports_double_buffering=True
            ),
            optimization=OptimizationConfig(
                default_algorithm="auto",
                cp_sat_time_limit=5.0,  # Shorter timeout for GPU
                cp_sat_enabled_threshold=32,
                greedy_threshold=500,
                enable_multi_matmul_fusion=True,
                enable_k_reduction_grouping=False  # GPU handles this in hardware
            )
        )

    @classmethod
    def default_tpu(cls) -> 'HardwareConfig':
        """Default TPU configuration"""
        return cls(
            name="TPU-v4",
            hardware_type=HardwareType.TPU,
            compute=ComputeConfig(
                batch_size=128,
                tile_size=(128, 128, 128),
                must_fill_batch=False,
                supports_variable_tiles=False,
                supports_mixed_precision=True,
                supported_dtypes=["bfloat16", "float32", "int8"],
                compute_throughput=100.0  # 100 TFLOPS
            ),
            memory=MemoryConfig(
                on_chip_size=16777216,  # 16MB vector memory
                ddr_bandwidth=1200.0,  # HBM bandwidth
                dma_channels=4,
                supports_double_buffering=True
            ),
            optimization=OptimizationConfig(
                default_algorithm="greedy",  # TPU prefers simple scheduling
                cp_sat_time_limit=2.0,
                cp_sat_enabled_threshold=10,  # Rarely use CP-SAT
                greedy_threshold=100,
                enable_multi_matmul_fusion=False,  # TPU compiler handles this
                enable_k_reduction_grouping=False
            )
        )

    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file"""
        data = self._to_dict()
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def to_json(self, json_path: str):
        """Save configuration to JSON file"""
        data = self._to_dict()
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

    def _to_dict(self) -> dict:
        """Convert to dictionary representation"""
        return {
            'hardware': {
                'name': self.name,
                'type': self.hardware_type.value,
                'vendor': self.vendor_extensions.get('vendor', 'generic')
            },
            'compute': {
                'batch_size': self.compute.batch_size,
                'tile_size': list(self.compute.tile_size),
                'must_fill_batch': self.compute.must_fill_batch,
                'supports_variable_tiles': self.compute.supports_variable_tiles,
                'supports_mixed_precision': self.compute.supports_mixed_precision,
                'supported_dtypes': self.compute.supported_dtypes,
                'compute_throughput': self.compute.compute_throughput
            },
            'memory': {
                'on_chip_buffer': self.memory.on_chip_size,
                'l2_cache': self.memory.l2_cache_size,
                'ddr_bandwidth': self.memory.ddr_bandwidth,
                'dma_channels': self.memory.dma_channels,
                'double_buffer': self.memory.supports_double_buffering
            },
            'optimization': {
                'default_algorithm': self.optimization.default_algorithm,
                'cp_sat_time_limit': self.optimization.cp_sat_time_limit,
                'cp_sat_enabled_threshold': self.optimization.cp_sat_enabled_threshold,
                'greedy_threshold': self.optimization.greedy_threshold,
                'enable_multi_matmul_fusion': self.optimization.enable_multi_matmul_fusion,
                'enable_k_reduction_grouping': self.optimization.enable_k_reduction_grouping
            },
            'vendor_extensions': self.vendor_extensions
        }

    def validate(self) -> bool:
        """Validate configuration consistency"""
        # Check batch size
        if self.compute.batch_size <= 0 or self.compute.batch_size > 256:
            return False

        # Check tile size
        for dim in self.compute.tile_size:
            if dim <= 0 or dim > 256:
                return False

        # Check memory
        if self.memory.on_chip_size <= 0:
            return False

        # Check thresholds
        if self.optimization.cp_sat_enabled_threshold >= self.optimization.greedy_threshold:
            return False

        return True