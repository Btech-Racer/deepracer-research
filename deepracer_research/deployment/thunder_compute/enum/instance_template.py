from enum import StrEnum
from typing import Dict, List


class InstanceTemplate(StrEnum):
    """Available instance templates for Thunder Compute instances"""

    BASE = "base"

    PYTORCH = "pytorch"

    TENSORFLOW = "tensorflow"

    CUDA = "cuda"

    @property
    def description(self) -> str:
        """Get the description of the instance template.

        Returns
        -------
        str
            Description of the template.
        """
        _descriptions = {
            self.BASE: "Basic Ubuntu environment with essential tools",
            self.PYTORCH: "PyTorch deep learning framework pre-installed",
            self.TENSORFLOW: "TensorFlow deep learning framework pre-installed",
            self.CUDA: "CUDA toolkit and drivers pre-installed for GPU computing",
        }
        return _descriptions[self]

    @property
    def included_packages(self) -> List[str]:
        """Get the list of included packages in the template.

        Returns
        -------
        List[str]
            List of major packages/frameworks included in the template.
        """
        _packages = {
            self.BASE: ["Ubuntu 20.04/22.04", "Python 3.8+", "Docker", "Git"],
            self.PYTORCH: ["Ubuntu 20.04/22.04", "Python 3.8+", "PyTorch", "CUDA", "Docker"],
            self.TENSORFLOW: ["Ubuntu 20.04/22.04", "Python 3.8+", "TensorFlow", "CUDA", "Docker"],
            self.CUDA: ["Ubuntu 20.04/22.04", "CUDA Toolkit", "NVIDIA Drivers", "Docker"],
        }
        return _packages[self]

    @property
    def is_ml_ready(self) -> bool:
        """Check if template is ready for machine learning workloads.

        Returns
        -------
        bool
            True if template includes ML frameworks.
        """
        return self in [self.PYTORCH, self.TENSORFLOW]

    @classmethod
    def get_recommended_for_deepracer(cls) -> "InstanceTemplate":
        """Get the recommended template for DeepRacer training.

        Returns
        -------
        InstanceTemplate
            Recommended template for DeepRacer workloads.
        """
        return cls.BASE

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, any]]:
        """Get comprehensive information about all instance templates.

        Returns
        -------
        Dict[str, Dict[str, any]]
            Dictionary mapping templates to their specifications.
        """
        return {
            template.value: {
                "description": template.description,
                "included_packages": template.included_packages,
                "ml_ready": template.is_ml_ready,
            }
            for template in cls
        }
