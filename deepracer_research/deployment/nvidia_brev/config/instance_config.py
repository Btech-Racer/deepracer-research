from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from deepracer_research.deployment.nvidia_brev.enum.deployment_mode import DeploymentMode
from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType
from deepracer_research.deployment.nvidia_brev.enum.instance_template import InstanceTemplate


@dataclass
class InstanceConfig:
    """Configuration for NVIDIA Brev instance creation

    Parameters
    ----------
    cpu_cores : int
        Number of CPU cores to allocate for the instance.
    template : InstanceTemplate
        Instance template defining the pre-installed software environment.
    gpu_type : GPUType
        Type of GPU to attach to the instance.
    deployment_mode : DeploymentMode, optional
        Deployment mode affecting pricing and availability, by default ON_DEMAND.
    num_gpus : int, optional
        Number of GPUs to attach, by default 1.
    memory_gb : int, optional
        RAM size in gigabytes, by default None (auto-calculated from GPU type).
    disk_size_gb : int, optional
        Disk size in gigabytes, by default 100.
    instance_name : str, optional
        Instance name/identifier. If not provided, auto-generated, by default None.
    region : str, optional
        Preferred region for deployment, by default None (auto-select).
    install_deepracer_cloud : bool, optional
        Whether to automatically install DeepRacer-for-Cloud, by default True.
    aws_credentials : Dict[str, str], optional
        AWS credentials for S3 access, by default empty dict.
    custom_setup_script : str, optional
        Custom setup script to run after instance creation, by default None.
    environment_variables : Dict[str, str], optional
        Environment variables to set on the instance, by default empty dict.
    ports : List[int], optional
        Ports to expose for external access, by default [8888, 6006] (Jupyter, TensorBoard).
    auto_shutdown_hours : int, optional
        Automatically shutdown instance after N hours of inactivity, by default None.
    """

    cpu_cores: int
    template: InstanceTemplate
    gpu_type: GPUType
    deployment_mode: DeploymentMode = DeploymentMode.ON_DEMAND
    num_gpus: int = 1
    memory_gb: Optional[int] = None
    disk_size_gb: int = 100
    instance_name: Optional[str] = None
    region: Optional[str] = None

    install_deepracer_cloud: bool = True
    aws_credentials: Dict[str, str] = field(default_factory=dict)

    custom_setup_script: Optional[str] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    ports: List[int] = field(default_factory=lambda: [8888, 6006])  # Jupyter, TensorBoard
    auto_shutdown_hours: Optional[int] = None

    def __post_init__(self):
        """Post-initialization validation and auto-configuration."""
        if self.memory_gb is None:
            self.memory_gb = self._calculate_recommended_memory()

        if self.instance_name is None:
            self.instance_name = self._generate_instance_name()

        self.validate()

    def _calculate_recommended_memory(self) -> int:
        """Calculate recommended memory based on GPU type and cores.

        Returns
        -------
        int
            Recommended memory in GB
        """
        base_memory = self.cpu_cores * 2

        gpu_memory_multiplier = {
            GPUType.H100: 8,
            GPUType.A100_80GB: 6,
            GPUType.A100: 4,
            GPUType.RTX_4090: 4,
            GPUType.RTX_6000_ADA: 4,
            GPUType.L40S: 4,
            GPUType.A10G: 3,
            GPUType.RTX_3090: 3,
            GPUType.RTX_4080: 2,
            GPUType.V100: 2,
            GPUType.T4: 2,
            GPUType.RTX_3080: 2,
        }.get(self.gpu_type, 2)

        gpu_memory = self.gpu_type.memory_gb * gpu_memory_multiplier * self.num_gpus

        return max(base_memory, gpu_memory, 16)

    def _generate_instance_name(self) -> str:
        """Generate a default instance name.

        Returns
        -------
        str
            Generated instance name
        """
        import uuid

        short_uuid = str(uuid.uuid4())[:8]
        return f"deepracer-{self.gpu_type.value}-{short_uuid}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for API calls.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation suitable for NVIDIA Brev API.
        """
        config = {
            "name": self.instance_name,
            "template": self.template.value,
            "gpu_type": self.gpu_type.value,
            "num_gpus": self.num_gpus,
            "cpu_cores": self.cpu_cores,
            "memory_gb": self.memory_gb,
            "disk_size_gb": self.disk_size_gb,
            "deployment_mode": self.deployment_mode.value,
            "ports": self.ports,
        }

        if self.region:
            config["region"] = self.region

        if self.auto_shutdown_hours:
            config["auto_shutdown_hours"] = self.auto_shutdown_hours

        if self.environment_variables:
            config["environment_variables"] = self.environment_variables

        if self.custom_setup_script:
            config["setup_script"] = self.custom_setup_script

        return config

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

        if self.memory_gb and self.memory_gb < 8:
            raise ValueError("Memory must be at least 8GB")

        if self.disk_size_gb < 20:
            raise ValueError("Disk size must be at least 20GB")

        if self.install_deepracer_cloud and self.disk_size_gb < 50:
            raise ValueError("DeepRacer installation requires at least 50GB disk space")

        if not self.gpu_type.is_suitable_for_training and self.install_deepracer_cloud:
            raise ValueError(f"GPU type {self.gpu_type.value} may not have sufficient memory for DeepRacer training")

        if self.auto_shutdown_hours is not None and self.auto_shutdown_hours <= 0:
            raise ValueError("Auto shutdown hours must be positive")

        for port in self.ports:
            if not (1 <= port <= 65535):
                raise ValueError(f"Invalid port number: {port}")

    @classmethod
    def for_deepracer_training(
        cls,
        cpu_cores: int = 8,
        gpu_type: GPUType = GPUType.A100,
        deployment_mode: DeploymentMode = DeploymentMode.SPOT,
        disk_size_gb: int = 200,
        auto_shutdown_hours: int = 8,
        **kwargs,
    ) -> "InstanceConfig":
        """Create an optimized configuration for DeepRacer training.

        Parameters
        ----------
        cpu_cores : int, optional
            Number of CPU cores, by default 8
        gpu_type : GPUType, optional
            GPU type for training, by default A100
        deployment_mode : DeploymentMode, optional
            Deployment mode, by default SPOT for cost savings
        disk_size_gb : int, optional
            Disk size in GB, by default 200
        auto_shutdown_hours : int, optional
            Auto shutdown after hours, by default 8
        **kwargs
            Additional configuration parameters

        Returns
        -------
        InstanceConfig
            Optimized configuration for DeepRacer training
        """
        return cls(
            cpu_cores=cpu_cores,
            template=InstanceTemplate.DEEPRACER_READY,
            gpu_type=gpu_type,
            deployment_mode=deployment_mode,
            disk_size_gb=disk_size_gb,
            install_deepracer_cloud=True,
            auto_shutdown_hours=auto_shutdown_hours,
            ports=[8888, 6006, 8080],
            environment_variables={"DEEPRACER_TRAINING": "true", "CUDA_VISIBLE_DEVICES": "0"},
            **kwargs,
        )

    @classmethod
    def for_development(
        cls,
        cpu_cores: int = 4,
        gpu_type: GPUType = GPUType.RTX_4080,
        template: InstanceTemplate = InstanceTemplate.JUPYTER_LAB,
        **kwargs,
    ) -> "InstanceConfig":
        """Create a configuration optimized for development work.

        Parameters
        ----------
        cpu_cores : int, optional
            Number of CPU cores, by default 4
        gpu_type : GPUType, optional
            GPU type, by default RTX_4080
        template : InstanceTemplate, optional
            Software template, by default JUPYTER_LAB
        **kwargs
            Additional configuration parameters

        Returns
        -------
        InstanceConfig
            Development-optimized configuration
        """
        return cls(
            cpu_cores=cpu_cores,
            template=template,
            gpu_type=gpu_type,
            deployment_mode=DeploymentMode.ON_DEMAND,
            disk_size_gb=100,
            install_deepracer_cloud=False,
            auto_shutdown_hours=4,
            ports=[8888, 6006, 8000],
            **kwargs,
        )

    def get_cost_estimate(self, hours: int) -> Dict[str, float]:
        """Estimate the cost for running this configuration.

        Parameters
        ----------
        hours : int
            Number of hours to run

        Returns
        -------
        Dict[str, float]
            Cost estimate breakdown
        """
        gpu_base_rates = {
            GPUType.H100: 4.00,
            GPUType.A100_80GB: 3.20,
            GPUType.A100: 2.40,
            GPUType.RTX_4090: 1.60,
            GPUType.RTX_6000_ADA: 2.00,
            GPUType.L40S: 1.80,
            GPUType.A10G: 1.20,
            GPUType.RTX_3090: 1.00,
            GPUType.RTX_4080: 0.80,
            GPUType.V100: 1.00,
            GPUType.T4: 0.60,
            GPUType.RTX_3080: 0.50,
        }

        base_hourly = gpu_base_rates.get(self.gpu_type, 1.0) * self.num_gpus
        cpu_hourly = self.cpu_cores * 0.05
        memory_hourly = self.memory_gb * 0.01
        storage_hourly = self.disk_size_gb * 0.001

        total_hourly = base_hourly + cpu_hourly + memory_hourly + storage_hourly

        adjusted_hourly = total_hourly * self.deployment_mode.cost_multiplier

        total_cost = adjusted_hourly * hours

        return {
            "gpu_cost": base_hourly * self.deployment_mode.cost_multiplier * hours,
            "cpu_cost": cpu_hourly * hours,
            "memory_cost": memory_hourly * hours,
            "storage_cost": storage_hourly * hours,
            "hourly_rate": adjusted_hourly,
            "total_cost": total_cost,
            "deployment_mode": self.deployment_mode.value,
            "cost_multiplier": self.deployment_mode.cost_multiplier,
        }
