from enum import Enum


class SageMakerInstanceType(str, Enum):
    """Supported instance types for SageMaker training"""

    ML_C5_LARGE = "ml.c5.large"
    ML_C5_XLARGE = "ml.c5.xlarge"
    ML_C5_2XLARGE = "ml.c5.2xlarge"
    ML_C5_4XLARGE = "ml.c5.4xlarge"
    ML_C5_9XLARGE = "ml.c5.9xlarge"
    ML_C5_18XLARGE = "ml.c5.18xlarge"

    ML_M5_LARGE = "ml.m5.large"
    ML_M5_XLARGE = "ml.m5.xlarge"
    ML_M5_2XLARGE = "ml.m5.2xlarge"
    ML_M5_4XLARGE = "ml.m5.4xlarge"
    ML_M5_12XLARGE = "ml.m5.12xlarge"
    ML_M5_24XLARGE = "ml.m5.24xlarge"

    ML_P3_2XLARGE = "ml.p3.2xlarge"
    ML_P3_8XLARGE = "ml.p3.8xlarge"
    ML_P3_16XLARGE = "ml.p3.16xlarge"

    ML_G4DN_XLARGE = "ml.g4dn.xlarge"
    ML_G4DN_2XLARGE = "ml.g4dn.2xlarge"
    ML_G4DN_4XLARGE = "ml.g4dn.4xlarge"
    ML_G4DN_8XLARGE = "ml.g4dn.8xlarge"
    ML_G4DN_12XLARGE = "ml.g4dn.12xlarge"
    ML_G4DN_16XLARGE = "ml.g4dn.16xlarge"

    def is_gpu_instance(self) -> bool:
        """Check if the instance type supports GPU.

        Returns
        -------
        bool
            True if instance has GPU support
        """
        return self.value.startswith(("ml.p3", "ml.g4dn"))

    def is_cpu_optimized(self) -> bool:
        """Check if the instance type is CPU-optimized.

        Returns
        -------
        bool
            True if instance is CPU-optimized
        """
        return self.value.startswith("ml.c5")

    def is_memory_optimized(self) -> bool:
        """Check if the instance type is memory-optimized.

        Returns
        -------
        bool
            True if instance is memory-optimized
        """
        return self.value.startswith("ml.m5")

    def get_size_category(self) -> str:
        """Get the size category of the instance.

        Returns
        -------
        str
            Size category: 'small', 'medium', 'large', or 'xlarge'
        """
        if "large" in self.value and not any(x in self.value for x in ["xlarge", "2xlarge"]):
            return "small"
        elif "xlarge" in self.value and not any(x in self.value for x in ["2xlarge", "4xlarge"]):
            return "medium"
        elif any(x in self.value for x in ["2xlarge", "4xlarge"]):
            return "large"
        else:
            return "xlarge"

    @classmethod
    def get_recommended_for_scenario(cls, scenario: str) -> "SageMakerInstanceType":
        """Get recommended instance type for a scenario.

        Parameters
        ----------
        scenario : str
            Training scenario

        Returns
        -------
        SageMakerInstanceType
            Recommended instance type
        """
        recommendations = {
            "development": cls.ML_C5_LARGE,
            "testing": cls.ML_C5_XLARGE,
            "production": cls.ML_C5_2XLARGE,
            "research": cls.ML_P3_2XLARGE,
            "large_scale": cls.ML_P3_8XLARGE,
        }
        return recommendations.get(scenario.lower(), cls.ML_C5_2XLARGE)

    def get_description(self) -> str:
        """Get  description of the instance type.

        Returns
        -------
        str
            Description of the instance type
        """
        descriptions = {
            self.ML_C5_LARGE: "Compute Optimized - Small (2 vCPU, 4 GB RAM)",
            self.ML_C5_XLARGE: "Compute Optimized - Medium (4 vCPU, 8 GB RAM)",
            self.ML_C5_2XLARGE: "Compute Optimized - Large (8 vCPU, 16 GB RAM)",
            self.ML_C5_4XLARGE: "Compute Optimized - XLarge (16 vCPU, 32 GB RAM)",
            self.ML_C5_9XLARGE: "Compute Optimized - 2XLarge (36 vCPU, 72 GB RAM)",
            self.ML_C5_18XLARGE: "Compute Optimized - 4XLarge (72 vCPU, 144 GB RAM)",
            self.ML_M5_LARGE: "General Purpose - Small (2 vCPU, 8 GB RAM)",
            self.ML_M5_XLARGE: "General Purpose - Medium (4 vCPU, 16 GB RAM)",
            self.ML_M5_2XLARGE: "General Purpose - Large (8 vCPU, 32 GB RAM)",
            self.ML_M5_4XLARGE: "General Purpose - XLarge (16 vCPU, 64 GB RAM)",
            self.ML_M5_12XLARGE: "General Purpose - 3XLarge (48 vCPU, 192 GB RAM)",
            self.ML_M5_24XLARGE: "General Purpose - 6XLarge (96 vCPU, 384 GB RAM)",
            self.ML_P3_2XLARGE: "GPU Accelerated - Medium (8 vCPU, 61 GB RAM, 1 V100 GPU)",
            self.ML_P3_8XLARGE: "GPU Accelerated - Large (32 vCPU, 244 GB RAM, 4 V100 GPU)",
            self.ML_P3_16XLARGE: "GPU Accelerated - XLarge (64 vCPU, 488 GB RAM, 8 V100 GPU)",
            self.ML_G4DN_XLARGE: "GPU Accelerated - Small (4 vCPU, 16 GB RAM, 1 T4 GPU)",
            self.ML_G4DN_2XLARGE: "GPU Accelerated - Medium (8 vCPU, 32 GB RAM, 1 T4 GPU)",
            self.ML_G4DN_4XLARGE: "GPU Accelerated - Large (16 vCPU, 64 GB RAM, 1 T4 GPU)",
            self.ML_G4DN_8XLARGE: "GPU Accelerated - XLarge (32 vCPU, 128 GB RAM, 1 T4 GPU)",
            self.ML_G4DN_12XLARGE: "GPU Accelerated - 2XLarge (48 vCPU, 192 GB RAM, 4 T4 GPU)",
            self.ML_G4DN_16XLARGE: "GPU Accelerated - 4XLarge (64 vCPU, 256 GB RAM, 1 T4 GPU)",
        }
        return descriptions.get(self, "Unknown instance type")
