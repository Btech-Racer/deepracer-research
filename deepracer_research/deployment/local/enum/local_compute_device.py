from enum import Enum
from typing import List, Optional


class LocalComputeDevice(str, Enum):
    """Supported compute devices for local training"""

    CPU = "cpu"

    GPU = "gpu"

    MPS = "mps"

    AUTO = "auto"

    @classmethod
    def get_available_devices(cls) -> List["LocalComputeDevice"]:
        """Get list of available compute devices on the current system.

        Returns
        -------
        List[LocalComputeDevice]
            List of available devices
        """
        available = [cls.CPU]

        try:
            import torch

            if torch.cuda.is_available():
                available.append(cls.GPU)
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                available.append(cls.MPS)
        except ImportError:
            pass

        try:
            import tensorflow as tf

            if len(tf.config.list_physical_devices("GPU")) > 0:
                if cls.GPU not in available:
                    available.append(cls.GPU)
        except ImportError:
            pass

        return available

    @classmethod
    def get_optimal_device(cls) -> "LocalComputeDevice":
        """Get the optimal device for the current system.

        Returns
        -------
        LocalComputeDevice
            Best available device for training (excludes AUTO)
        """
        available = cls.get_available_devices()

        if cls.GPU in available:
            return cls.GPU
        elif cls.MPS in available:
            return cls.MPS
        else:
            return cls.CPU

    def resolve_auto_device(self) -> "LocalComputeDevice":
        """Resolve AUTO device to the actual optimal device.

        Returns
        -------
        LocalComputeDevice
            The actual device to use (never AUTO)
        """
        if self == self.AUTO:
            return self.get_optimal_device()
        return self

    def get_memory_info(self) -> Optional[dict]:
        """Get memory information for the device.

        Returns
        -------
        Optional[dict]
            Memory information if available
        """
        if self == self.AUTO:
            resolved_device = self.resolve_auto_device()
            return resolved_device.get_memory_info()

        if self == self.CPU:
            try:
                import psutil

                return {
                    "total_memory_gb": psutil.virtual_memory().total / (1024**3),
                    "available_memory_gb": psutil.virtual_memory().available / (1024**3),
                }
            except ImportError:
                return None

        elif self == self.GPU:
            try:
                import torch

                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    return {
                        "device_count": device_count,
                        "total_memory_gb": torch.cuda.get_device_properties(0).total_memory / (1024**3),
                        "device_name": torch.cuda.get_device_name(0),
                    }
            except ImportError:
                pass

            try:
                import nvidia_ml_py3 as nvml

                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "total_memory_gb": info.total / (1024**3),
                    "free_memory_gb": info.free / (1024**3),
                    "used_memory_gb": info.used / (1024**3),
                }
            except ImportError:
                return None

        return None

    def is_available(self) -> bool:
        """Check if this device is available on the current system.

        Returns
        -------
        bool
            True if device is available
        """
        if self == self.AUTO:
            return True
        return self in self.get_available_devices()

    def get_description(self) -> str:
        """Get a  description of the device.

        Returns
        -------
        str
            Description of the compute device
        """
        descriptions = {
            self.CPU: "CPU - Universal compatibility, slower training",
            self.GPU: "GPU (CUDA) - Fast training, requires NVIDIA GPU",
            self.MPS: "MPS - Apple Silicon acceleration for M1/M2/M3 Macs",
            self.AUTO: "AUTO - Automatically selects the best available device",
        }
        return descriptions[self]
