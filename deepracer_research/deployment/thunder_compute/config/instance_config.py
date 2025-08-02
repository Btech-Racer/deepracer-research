from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from deepracer_research.deployment.thunder_compute.enum.deployment_mode import DeploymentMode
from deepracer_research.deployment.thunder_compute.enum.gpu_type import GPUType
from deepracer_research.deployment.thunder_compute.enum.instance_template import InstanceTemplate


@dataclass
class InstanceConfig:
    """Configuration for Thunder Compute instance creation

    Parameters
    ----------
    cpu_cores : int
        Number of CPU cores to allocate for the instance.
    template : InstanceTemplate
        Instance template defining the pre-installed software environment.
    gpu_type : GPUType
        Type of GPU to attach to the instance.
    num_gpus : int, optional
        Number of GPUs to attach, by default 1.
    disk_size_gb : int, optional
        Disk size in gigabytes, by default 100.
    mode : DeploymentMode, optional
        Deployment mode affecting pricing and availability, by default PRODUCTION.
    identifier : str, optional
        Instance identifier/name. If not provided, auto-generated, by default None.
    install_deepracer_cloud : bool, optional
        Whether to automatically install DeepRacer-for-Cloud, by default True.
    s3_bucket_name : str, optional
        S3 bucket name for model storage, by default None.
    aws_profile : str, optional
        AWS profile name for S3 access, by default "default".
    custom_setup_script : str, optional
        Custom setup script to run after instance creation, by default None.
    environment_variables : Dict[str, str], optional
        Environment variables to set on the instance, by default empty dict.
    """

    cpu_cores: int
    template: InstanceTemplate
    gpu_type: GPUType
    num_gpus: int = 1
    disk_size_gb: int = 100
    mode: DeploymentMode = DeploymentMode.PRODUCTION
    identifier: Optional[str] = None

    install_deepracer_cloud: bool = True
    s3_bucket_name: Optional[str] = None
    aws_profile: str = "default"

    custom_setup_script: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for API calls.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for Thunder Compute API.
        """
        return {
            "cpu_cores": self.cpu_cores,
            "template": self.template.value,
            "gpu_type": self.gpu_type.value,
            "num_gpus": self.num_gpus,
            "disk_size_gb": self.disk_size_gb,
            "mode": self.mode.value,
        }

    def validate(self) -> None:
        """Validate the configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if self.cpu_cores <= 0:
            raise ValueError("CPU cores must be positive")

        if self.num_gpus <= 0:
            raise ValueError("Number of GPUs must be positive")

        if self.disk_size_gb < 20:
            raise ValueError("Disk size must be at least 20GB")

        if self.install_deepracer_cloud and self.disk_size_gb < 40:
            raise ValueError("DeepRacer installation requires at least 40GB disk space")

        if not self.gpu_type.is_suitable_for_training and self.install_deepracer_cloud:
            raise ValueError(f"GPU type {self.gpu_type.value} may not have sufficient memory for DeepRacer training")

    @classmethod
    def for_deepracer_training(
        cls, cpu_cores: int = 8, gpu_type: GPUType = GPUType.A100, disk_size_gb: int = 100, s3_bucket_name: Optional[str] = None
    ) -> "InstanceConfig":
        """Create configuration optimized for DeepRacer training.

        Parameters
        ----------
        cpu_cores : int, optional
            Number of CPU cores, by default 8 (recommended for training).
        gpu_type : GPUType, optional
            GPU type for training, by default T4.
        disk_size_gb : int, optional
            Disk size in GB, by default 100 (minimum 40GB for DeepRacer).
        s3_bucket_name : str, optional
            S3 bucket for model storage, by default None.

        Returns
        -------
        InstanceConfig
            Instance configuration optimized for DeepRacer training.

        Raises
        ------
        ValueError
            If disk_size_gb is less than 40GB.
        """
        if disk_size_gb < 40:
            raise ValueError("DeepRacer requires minimum 40GB disk space")

        return cls(
            cpu_cores=cpu_cores,
            template=InstanceTemplate.BASE,
            gpu_type=gpu_type,
            num_gpus=1,
            disk_size_gb=disk_size_gb,
            mode=DeploymentMode.PRODUCTION,
            install_deepracer_cloud=True,
            s3_bucket_name=s3_bucket_name,
            environment_variables={
                "DR_CLOUD": "local",
                "DR_LOCAL_S3_PROFILE": "default",
                "DR_LOCAL_S3_BUCKET": s3_bucket_name or "",
                "DR_UPLOAD_S3_PROFILE": "default",
                "DR_UPLOAD_S3_BUCKET": s3_bucket_name or "",
            },
        )

    @classmethod
    def for_deepracer_evaluation(
        cls, cpu_cores: int = 4, gpu_type: GPUType = GPUType.T4, disk_size_gb: int = 50
    ) -> "InstanceConfig":
        """Create configuration optimized for DeepRacer evaluation.

        Parameters
        ----------
        cpu_cores : int, optional
            Number of CPU cores, by default 4.
        gpu_type : GPUType, optional
            GPU type for evaluation, by default T4.
        disk_size_gb : int, optional
            Disk size in GB, by default 50.

        Returns
        -------
        InstanceConfig
            Instance configuration optimized for DeepRacer evaluation.
        """
        return cls(
            cpu_cores=cpu_cores,
            template=InstanceTemplate.BASE,
            gpu_type=gpu_type,
            num_gpus=1,
            disk_size_gb=disk_size_gb,
            mode=DeploymentMode.PRODUCTION,
            install_deepracer_cloud=True,
        )

    @classmethod
    def for_research(
        cls, cpu_cores: int = 16, gpu_type: GPUType = GPUType.A100, num_gpus: int = 2, disk_size_gb: int = 200
    ) -> "InstanceConfig":
        """Create configuration optimized for research workloads.

        Parameters
        ----------
        cpu_cores : int, optional
            Number of CPU cores, by default 16.
        gpu_type : GPUType, optional
            GPU type for research, by default A100.
        num_gpus : int, optional
            Number of GPUs, by default 2.
        disk_size_gb : int, optional
            Disk size in GB, by default 200.

        Returns
        -------
        InstanceConfig
            Instance configuration optimized for research workloads.
        """
        return cls(
            cpu_cores=cpu_cores,
            template=InstanceTemplate.BASE,
            gpu_type=gpu_type,
            num_gpus=num_gpus,
            disk_size_gb=disk_size_gb,
            mode=DeploymentMode.PRODUCTION,
            install_deepracer_cloud=True,
            environment_variables={
                "DR_WORKERS": str(min(4, cpu_cores // 4)),
                "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(num_gpus)),
            },
        )
