from enum import StrEnum
from typing import Any, Dict, List


class EC2InstanceType(StrEnum):
    """Available EC2 instance types for DeepRacer deployments"""

    G4DN_XLARGE = "g4dn.xlarge"
    G4DN_2XLARGE = "g4dn.2xlarge"
    G4DN_4XLARGE = "g4dn.4xlarge"

    P3_2XLARGE = "p3.2xlarge"
    P3_8XLARGE = "p3.8xlarge"

    P4D_24XLARGE = "p4d.24xlarge"

    C5_LARGE = "c5.large"
    C5_XLARGE = "c5.xlarge"
    C5_2XLARGE = "c5.2xlarge"

    @property
    def display_name(self) -> str:
        """Get the display name for the instance type.

        Returns
        -------
        str
            Display name for the instance type.
        """
        _display_names = {
            self.G4DN_XLARGE: "g4dn.xlarge (1x NVIDIA T4, 4 vCPU, 16 GB)",
            self.G4DN_2XLARGE: "g4dn.2xlarge (1x NVIDIA T4, 8 vCPU, 32 GB)",
            self.G4DN_4XLARGE: "g4dn.4xlarge (1x NVIDIA T4, 16 vCPU, 64 GB)",
            self.P3_2XLARGE: "p3.2xlarge (1x NVIDIA V100, 8 vCPU, 61 GB)",
            self.P3_8XLARGE: "p3.8xlarge (4x NVIDIA V100, 32 vCPU, 244 GB)",
            self.P4D_24XLARGE: "p4d.24xlarge (8x NVIDIA A100, 96 vCPU, 1152 GB)",
            self.C5_LARGE: "c5.large (2 vCPU, 4 GB RAM)",
            self.C5_XLARGE: "c5.xlarge (4 vCPU, 8 GB RAM)",
            self.C5_2XLARGE: "c5.2xlarge (8 vCPU, 16 GB RAM)",
        }
        return _display_names[self]

    @property
    def vcpus(self) -> int:
        """Get the number of vCPUs.

        Returns
        -------
        int
            Number of virtual CPUs.
        """
        _vcpus = {
            self.G4DN_XLARGE: 4,
            self.G4DN_2XLARGE: 8,
            self.G4DN_4XLARGE: 16,
            self.P3_2XLARGE: 8,
            self.P3_8XLARGE: 32,
            self.P4D_24XLARGE: 96,
            self.C5_LARGE: 2,
            self.C5_XLARGE: 4,
            self.C5_2XLARGE: 8,
        }
        return _vcpus[self]

    @property
    def memory_gb(self) -> int:
        """Get the memory size in GB.

        Returns
        -------
        int
            Memory size in gigabytes.
        """
        _memory_sizes = {
            self.G4DN_XLARGE: 16,
            self.G4DN_2XLARGE: 32,
            self.G4DN_4XLARGE: 64,
            self.P3_2XLARGE: 61,
            self.P3_8XLARGE: 244,
            self.P4D_24XLARGE: 1152,
            self.C5_LARGE: 4,
            self.C5_XLARGE: 8,
            self.C5_2XLARGE: 16,
        }
        return _memory_sizes[self]

    @property
    def gpu_count(self) -> int:
        """Get the number of GPUs.

        Returns
        -------
        int
            Number of GPUs (0 for CPU-only instances).
        """
        _gpu_counts = {
            self.G4DN_XLARGE: 1,
            self.G4DN_2XLARGE: 1,
            self.G4DN_4XLARGE: 1,
            self.P3_2XLARGE: 1,
            self.P3_8XLARGE: 4,
            self.P4D_24XLARGE: 8,
            self.C5_LARGE: 0,
            self.C5_XLARGE: 0,
            self.C5_2XLARGE: 0,
        }
        return _gpu_counts[self]

    @property
    def gpu_type(self) -> str:
        """Get the GPU type.

        Returns
        -------
        str
            GPU type name (empty for CPU-only instances).
        """
        _gpu_types = {
            self.G4DN_XLARGE: "NVIDIA T4",
            self.G4DN_2XLARGE: "NVIDIA T4",
            self.G4DN_4XLARGE: "NVIDIA T4",
            self.P3_2XLARGE: "NVIDIA V100",
            self.P3_8XLARGE: "NVIDIA V100",
            self.P4D_24XLARGE: "NVIDIA A100",
            self.C5_LARGE: "",
            self.C5_XLARGE: "",
            self.C5_2XLARGE: "",
        }
        return _gpu_types[self]

    @property
    def has_gpu(self) -> bool:
        """Check if instance type has GPU.

        Returns
        -------
        bool
            True if instance has GPU acceleration.
        """
        return self.gpu_count > 0

    @property
    def is_suitable_for_training(self) -> bool:
        """Check if instance is suitable for DeepRacer training.

        Returns
        -------
        bool
            True if instance has sufficient resources for training.
        """
        return self.has_gpu and self.memory_gb >= 16

    @property
    def is_suitable_for_evaluation(self) -> bool:
        """Check if instance is suitable for evaluation.

        Returns
        -------
        bool
            True if instance has sufficient resources for evaluation.
        """
        return self.memory_gb >= 4

    @property
    def hourly_cost_estimate(self) -> float:
        """Get approximate hourly cost in USD.

        Returns
        -------
        float
            Estimated hourly cost (subject to change).
        """
        _costs = {
            self.G4DN_XLARGE: 0.526,
            self.G4DN_2XLARGE: 0.752,
            self.G4DN_4XLARGE: 1.204,
            self.P3_2XLARGE: 3.06,
            self.P3_8XLARGE: 12.24,
            self.P4D_24XLARGE: 32.77,
            self.C5_LARGE: 0.085,
            self.C5_XLARGE: 0.17,
            self.C5_2XLARGE: 0.34,
        }
        return _costs[self]

    @classmethod
    def get_recommended_for_training(cls) -> "EC2InstanceType":
        """Get the recommended instance type for DeepRacer training.

        Returns
        -------
        EC2InstanceType
            Recommended instance type balancing cost and performance.
        """
        return cls.G4DN_XLARGE

    @classmethod
    def get_recommended_for_evaluation(cls) -> "EC2InstanceType":
        """Get the recommended instance type for evaluation.

        Returns
        -------
        EC2InstanceType
            Recommended instance type for evaluation tasks.
        """
        return cls.C5_XLARGE

    @classmethod
    def get_training_instances(cls) -> List["EC2InstanceType"]:
        """Get all instance types suitable for training.

        Returns
        -------
        List[EC2InstanceType]
            List of training-suitable instance types.
        """
        return [inst for inst in cls if inst.is_suitable_for_training]

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all instance types.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping instance types to their specifications.
        """
        return {
            instance_type.value: {
                "display_name": instance_type.display_name,
                "vcpus": instance_type.vcpus,
                "memory_gb": instance_type.memory_gb,
                "gpu_count": instance_type.gpu_count,
                "gpu_type": instance_type.gpu_type,
                "has_gpu": instance_type.has_gpu,
                "suitable_for_training": instance_type.is_suitable_for_training,
                "suitable_for_evaluation": instance_type.is_suitable_for_evaluation,
                "hourly_cost_estimate": instance_type.hourly_cost_estimate,
            }
            for instance_type in cls
        }
