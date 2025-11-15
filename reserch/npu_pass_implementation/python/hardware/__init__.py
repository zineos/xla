"""
硬件配置模块 - 各种硬件平台的预定义配置
"""

from ..core.hardware_config import HardwareConfig, HardwareType

__all__ = [
    "get_npu_config",
    "get_gpu_config",
    "get_tpu_config",
]


def get_npu_config(config_path: str = None) -> HardwareConfig:
    """
    获取NPU硬件配置

    Args:
        config_path: YAML配置文件路径，None则使用默认配置

    Returns:
        HardwareConfig
    """
    if config_path:
        return HardwareConfig.from_yaml(config_path)
    else:
        return HardwareConfig.get_default(HardwareType.NPU)


def get_gpu_config(config_path: str = None) -> HardwareConfig:
    """
    获取GPU硬件配置

    Args:
        config_path: YAML配置文件路径，None则使用默认配置

    Returns:
        HardwareConfig
    """
    if config_path:
        return HardwareConfig.from_yaml(config_path)
    else:
        return HardwareConfig.get_default(HardwareType.GPU)


def get_tpu_config(config_path: str = None) -> HardwareConfig:
    """
    获取TPU硬件配置

    Args:
        config_path: YAML配置文件路径，None则使用默认配置

    Returns:
        HardwareConfig
    """
    if config_path:
        return HardwareConfig.from_yaml(config_path)
    else:
        return HardwareConfig.get_default(HardwareType.TPU)
