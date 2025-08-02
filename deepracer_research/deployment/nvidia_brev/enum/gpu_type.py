from enum import StrEnum
from typing import Any, Dict


class GPUType(StrEnum):
    """Available GPU types for NVIDIA Brev instances"""

    RTX_3080 = "rtx-3080"
    RTX_3090 = "rtx-3090"
    RTX_4080 = "rtx-4080"
    RTX_4090 = "rtx-4090"

    A10G = "a10g"
    A100 = "a100"
    A100_80GB = "a100-80gb"
    H100 = "h100"
    V100 = "v100"
    T4 = "t4"

    RTX_6000_ADA = "rtx-6000-ada"
    L40S = "l40s"

    @property
    def display_name(self) -> str:
        """Get the display name for the GPU type

        Returns
        -------
        str
            Display name for the GPU type
        """
        _display_names = {
            self.RTX_3080: "NVIDIA GeForce RTX 3080",
            self.RTX_3090: "NVIDIA GeForce RTX 3090",
            self.RTX_4080: "NVIDIA GeForce RTX 4080",
            self.RTX_4090: "NVIDIA GeForce RTX 4090",
            self.A10G: "NVIDIA A10G",
            self.A100: "NVIDIA A100 40GB",
            self.A100_80GB: "NVIDIA A100 80GB",
            self.H100: "NVIDIA H100",
            self.V100: "NVIDIA V100",
            self.T4: "NVIDIA Tesla T4",
            self.RTX_6000_ADA: "NVIDIA RTX 6000 Ada Generation",
            self.L40S: "NVIDIA L40S",
        }
        return _display_names[self]

    @property
    def memory_gb(self) -> int:
        """Get the GPU memory in GB

        Returns
        -------
        int
            GPU memory size in gigabytes
        """
        _memory_sizes = {
            self.RTX_3080: 10,
            self.RTX_3090: 24,
            self.RTX_4080: 16,
            self.RTX_4090: 24,
            self.A10G: 24,
            self.A100: 40,
            self.A100_80GB: 80,
            self.H100: 80,
            self.V100: 16,
            self.T4: 16,
            self.RTX_6000_ADA: 48,
            self.L40S: 48,
        }
        return _memory_sizes[self]

    @property
    def is_suitable_for_training(self) -> bool:
        """Check if GPU is suitable for DeepRacer training.

        Returns
        -------
        bool
            True if GPU has sufficient memory and compute capability
        """
        return self.memory_gb >= 16

    @property
    def compute_capability(self) -> float:
        """Get the CUDA compute capability.

        Returns
        -------
        float
            CUDA compute capability version
        """
        _compute_capabilities = {
            self.RTX_3080: 8.6,
            self.RTX_3090: 8.6,
            self.RTX_4080: 8.9,
            self.RTX_4090: 8.9,
            self.A10G: 8.6,
            self.A100: 8.0,
            self.A100_80GB: 8.0,
            self.H100: 9.0,
            self.V100: 7.0,
            self.T4: 7.5,
            self.RTX_6000_ADA: 8.9,
            self.L40S: 8.9,
        }
        return _compute_capabilities[self]

    @property
    def is_datacenter_gpu(self) -> bool:
        """Check if this is a datacenter/professional GPU.

        Returns
        -------
        bool
            True if this is a datacenter GPU
        """
        datacenter_gpus = {self.A10G, self.A100, self.A100_80GB, self.H100, self.V100, self.T4, self.RTX_6000_ADA, self.L40S}
        return self in datacenter_gpus

    def get_recommended_cpu_cores(self) -> int:
        """Get recommended CPU cores for this GPU type.

        Returns
        -------
        int
            Recommended number of CPU cores
        """
        if self in {self.H100, self.A100_80GB}:
            return 16
        elif self in {self.A100, self.RTX_4090, self.RTX_6000_ADA, self.L40S}:
            return 12
        elif self in {self.A10G, self.RTX_3090, self.RTX_4080}:
            return 8
        else:
            return 4

    def to_dict(self) -> Dict[str, Any]:
        """Convert GPU type information to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of GPU type information
        """
        return {
            "type": self.value,
            "display_name": self.display_name,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability,
            "is_datacenter": self.is_datacenter_gpu,
            "is_suitable_for_training": self.is_suitable_for_training,
            "recommended_cpu_cores": self.get_recommended_cpu_cores(),
        }
