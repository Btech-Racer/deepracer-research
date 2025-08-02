from enum import StrEnum
from typing import Dict, List


class InstanceTemplate(StrEnum):
    """Available instance templates for NVIDIA Brev"""

    UBUNTU_20_04_CUDA = "ubuntu-20.04-cuda"
    UBUNTU_22_04_CUDA = "ubuntu-22.04-cuda"
    UBUNTU_24_04_CUDA = "ubuntu-24.04-cuda"

    PYTORCH_LATEST = "pytorch-latest"
    PYTORCH_2_1 = "pytorch-2.1"
    PYTORCH_2_0 = "pytorch-2.0"
    TENSORFLOW_LATEST = "tensorflow-latest"
    TENSORFLOW_2_13 = "tensorflow-2.13"
    TENSORFLOW_2_12 = "tensorflow-2.12"

    JUPYTER_LAB = "jupyter-lab"
    VSCODE_SERVER = "vscode-server"

    HUGGINGFACE_TRANSFORMERS = "huggingface-transformers"
    RAPIDS_AI = "rapids-ai"
    NVIDIA_TRITON = "nvidia-triton"

    DEEPRACER_READY = "deepracer-ready"

    @property
    def display_name(self) -> str:
        """Get  display name.

        Returns
        -------
        str
            Display name for the template
        """
        _display_names = {
            self.UBUNTU_20_04_CUDA: "Ubuntu 20.04 with CUDA",
            self.UBUNTU_22_04_CUDA: "Ubuntu 22.04 with CUDA",
            self.UBUNTU_24_04_CUDA: "Ubuntu 24.04 with CUDA",
            self.PYTORCH_LATEST: "PyTorch Latest",
            self.PYTORCH_2_1: "PyTorch 2.1",
            self.PYTORCH_2_0: "PyTorch 2.0",
            self.TENSORFLOW_LATEST: "TensorFlow Latest",
            self.TENSORFLOW_2_13: "TensorFlow 2.13",
            self.TENSORFLOW_2_12: "TensorFlow 2.12",
            self.JUPYTER_LAB: "JupyterLab Environment",
            self.VSCODE_SERVER: "VS Code Server Environment",
            self.HUGGINGFACE_TRANSFORMERS: "Hugging Face Transformers",
            self.RAPIDS_AI: "RAPIDS AI",
            self.NVIDIA_TRITON: "NVIDIA Triton Inference Server",
            self.DEEPRACER_READY: "DeepRacer Training Ready",
        }
        return _display_names[self]

    @property
    def description(self) -> str:
        """Get detailed description of the template.

        Returns
        -------
        str
            Detailed description
        """
        _descriptions = {
            self.UBUNTU_20_04_CUDA: "Ubuntu 20.04 LTS with NVIDIA CUDA drivers and toolkit",
            self.UBUNTU_22_04_CUDA: "Ubuntu 22.04 LTS with NVIDIA CUDA drivers and toolkit",
            self.UBUNTU_24_04_CUDA: "Ubuntu 24.04 LTS with NVIDIA CUDA drivers and toolkit",
            self.PYTORCH_LATEST: "Latest PyTorch with CUDA support for deep learning",
            self.PYTORCH_2_1: "PyTorch 2.1 with CUDA support for deep learning",
            self.PYTORCH_2_0: "PyTorch 2.0 with CUDA support for deep learning",
            self.TENSORFLOW_LATEST: "Latest TensorFlow with CUDA support",
            self.TENSORFLOW_2_13: "TensorFlow 2.13 with CUDA support",
            self.TENSORFLOW_2_12: "TensorFlow 2.12 with CUDA support",
            self.JUPYTER_LAB: "JupyterLab with common ML libraries and CUDA support",
            self.VSCODE_SERVER: "VS Code Server with GPU development tools",
            self.HUGGINGFACE_TRANSFORMERS: "Hugging Face Transformers library with CUDA",
            self.RAPIDS_AI: "RAPIDS AI libraries for GPU-accelerated data science",
            self.NVIDIA_TRITON: "NVIDIA Triton Inference Server for model deployment",
            self.DEEPRACER_READY: "Pre-configured environment for AWS DeepRacer training",
        }
        return _descriptions[self]

    @property
    def base_os(self) -> str:
        """Get the base operating system.

        Returns
        -------
        str
            Base operating system
        """
        if "ubuntu-20.04" in self.value:
            return "ubuntu-20.04"
        elif "ubuntu-22.04" in self.value:
            return "ubuntu-22.04"
        elif "ubuntu-24.04" in self.value:
            return "ubuntu-24.04"
        else:
            return "ubuntu-22.04"

    @property
    def has_cuda(self) -> bool:
        """Check if template includes CUDA support.

        Returns
        -------
        bool
            True if CUDA is included
        """
        return True

    @property
    def pre_installed_packages(self) -> List[str]:
        """Get list of pre-installed packages.

        Returns
        -------
        List[str]
            List of pre-installed packages
        """
        base_packages = ["nvidia-driver", "cuda-toolkit", "docker", "git", "python3", "pip"]

        package_map = {
            self.PYTORCH_LATEST: base_packages + ["pytorch", "torchvision", "torchaudio"],
            self.PYTORCH_2_1: base_packages + ["pytorch==2.1.*", "torchvision", "torchaudio"],
            self.PYTORCH_2_0: base_packages + ["pytorch==2.0.*", "torchvision", "torchaudio"],
            self.TENSORFLOW_LATEST: base_packages + ["tensorflow", "tensorflow-gpu"],
            self.TENSORFLOW_2_13: base_packages + ["tensorflow==2.13.*"],
            self.TENSORFLOW_2_12: base_packages + ["tensorflow==2.12.*"],
            self.JUPYTER_LAB: base_packages + ["jupyterlab", "numpy", "pandas", "matplotlib"],
            self.VSCODE_SERVER: base_packages + ["code-server"],
            self.HUGGINGFACE_TRANSFORMERS: base_packages + ["transformers", "datasets", "tokenizers"],
            self.RAPIDS_AI: base_packages + ["cudf", "cuml", "cugraph", "cuspatial"],
            self.NVIDIA_TRITON: base_packages + ["tritonserver"],
            self.DEEPRACER_READY: base_packages + ["tensorflow", "boto3", "deepracer-for-cloud"],
        }

        return package_map.get(self, base_packages)

    @property
    def is_suitable_for_deepracer(self) -> bool:
        """Check if template is suitable for DeepRacer training.

        Returns
        -------
        bool
            True if suitable for DeepRacer
        """
        suitable_templates = {
            self.PYTORCH_LATEST,
            self.PYTORCH_2_1,
            self.PYTORCH_2_0,
            self.TENSORFLOW_LATEST,
            self.TENSORFLOW_2_13,
            self.TENSORFLOW_2_12,
            self.JUPYTER_LAB,
            self.UBUNTU_22_04_CUDA,
            self.UBUNTU_24_04_CUDA,
            self.DEEPRACER_READY,
        }
        return self in suitable_templates

    @classmethod
    def get_recommended_for_deepracer(cls) -> "InstanceTemplate":
        """Get the recommended template for DeepRacer training.

        Returns
        -------
        InstanceTemplate
            Recommended template
        """
        return cls.DEEPRACER_READY

    def to_dict(self) -> Dict[str, any]:
        """Convert template information to dictionary.

        Returns
        -------
        Dict[str, any]
            Dictionary representation
        """
        return {
            "template": self.value,
            "display_name": self.display_name,
            "description": self.description,
            "base_os": self.base_os,
            "has_cuda": self.has_cuda,
            "pre_installed_packages": self.pre_installed_packages,
            "is_suitable_for_deepracer": self.is_suitable_for_deepracer,
        }
