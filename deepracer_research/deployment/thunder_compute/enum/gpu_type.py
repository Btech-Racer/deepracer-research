from enum import StrEnum
from typing import Any, Dict


class GPUType(StrEnum):
    """Available GPU types for Thunder Compute instances"""

    T4 = "t4"

    A100 = "a100"

    A100_XL = "a100xl"

    @property
    def display_name(self) -> str:
        """Get the display name for the GPU type.

        Returns
        -------
        str
            Display name for the GPU type.
        """
        _display_names = {
            self.T4: "NVIDIA Tesla T4",
            self.A100: "NVIDIA A100",
            self.A100_XL: "NVIDIA A100 XL",
        }
        return _display_names[self]

    @property
    def memory_gb(self) -> int:
        """Get the GPU memory in GB.

        Returns
        -------
        int
            GPU memory size in gigabytes.
        """
        _memory_sizes = {
            self.T4: 16,
            self.A100: 80,
            self.A100_XL: 80,
        }
        return _memory_sizes[self]

    @property
    def is_suitable_for_training(self) -> bool:
        """Check if GPU is suitable for DeepRacer training.

        Returns
        -------
        bool
            True if GPU has sufficient memory and performance for training.
        """
        return self.memory_gb >= 16

    @classmethod
    def get_recommended_for_training(cls) -> "GPUType":
        """Get the recommended GPU type for DeepRacer training.

        Returns
        -------
        GPUType
            Recommended GPU type balancing cost and performance.
        """
        return cls.T4

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all GPU types.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping GPU types to their specifications.
        """
        return {
            gpu_type.value: {
                "display_name": gpu_type.display_name,
                "memory_gb": gpu_type.memory_gb,
                "suitable_for_training": gpu_type.is_suitable_for_training,
            }
            for gpu_type in cls
        }
