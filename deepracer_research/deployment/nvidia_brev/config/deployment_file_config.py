from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from deepracer_research.deployment.deployment_target import DeploymentTarget
from deepracer_research.deployment.generators import DeepRacerConfigGenerator
from deepracer_research.deployment.nvidia_brev.enum.deployment_mode import DeploymentMode
from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario


@dataclass
class NvidiaBrevDeploymentFileConfig:
    """Configuration for generating deployment files for NVIDIA Brev

    Parameters
    ----------
    model_name : str
        Name of the DeepRacer model
    experimental_scenario : ExperimentalScenario
        Experimental scenario for training
    reward_function_code : str
        Python code for the reward function
    api_token : str
        NVIDIA Brev API token
    gpu_type : GPUType, optional
        GPU type for training, by default A100
    deployment_mode : DeploymentMode, optional
        Deployment mode, by default SPOT
    region : str, optional
        AWS region for S3, by default "us-east-1"
    s3_bucket : Optional[str], optional
        S3 bucket name, by default None (will create new)
    create_s3_bucket : bool, optional
        Whether to create a new S3 bucket, by default True
    output_dir : Optional[str], optional
        Output directory for generated files, by default None
    """

    model_name: str
    experimental_scenario: ExperimentalScenario
    reward_function_code: str
    api_token: str
    gpu_type: GPUType = GPUType.A100
    deployment_mode: DeploymentMode = DeploymentMode.SPOT
    region: str = "us-east-1"
    s3_bucket: Optional[str] = None
    create_s3_bucket: bool = True
    output_dir: Optional[str] = None

    def generate_deployment_files(self) -> Dict[str, Any]:
        """Generate all deployment files for NVIDIA Brev.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing file paths and deployment information
        """
        if self.output_dir is None:
            self.output_dir = Path.cwd() / "nvidia_brev_deployment_files"

        generator = DeepRacerConfigGenerator(output_dir=self.output_dir)

        result = generator.generate_all_files(
            model_name=self.model_name,
            deployment_target=DeploymentTarget.NVIDIA_BREV,
            experimental_scenario=self.experimental_scenario,
            reward_function_code=self.reward_function_code,
            s3_bucket=self.s3_bucket,
            create_s3_bucket=self.create_s3_bucket,
            gpu_type=self.gpu_type.value,
            deployment_mode=self.deployment_mode.value,
            api_token=self.api_token,
            region=self.region,
            cloud=DeploymentTarget.NVIDIA_BREV,
            workers=4,
            gui_enable=True,
            host_x=True,
            display=":99",
            robomaker_mount_logs=True,
            camera_sub_enable=True,
            eval_save_mp4=True,
            robomaker_cuda_devices="0",
            sagemaker_cuda_devices="0",
        )

        result["deployment_instructions"] = self._generate_brev_instructions(result)
        result["cost_estimate"] = self._get_cost_estimate()

        return result

    def _get_cost_estimate(self) -> Dict[str, float]:
        """Get cost estimate for NVIDIA Brev deployment."""
        from deepracer_research.deployment.nvidia_brev import get_cost_estimate

        return get_cost_estimate(gpu_type=self.gpu_type, deployment_mode=self.deployment_mode, hours=8)

    def _generate_brev_instructions(self, result: Dict[str, Any]) -> str:
        """Generate deployment instructions for NVIDIA Brev."""
        s3_bucket = result.get("s3_info", {}).get("name", self.s3_bucket)
        cost_estimate = self._get_cost_estimate()

        instructions = f"""
# NVIDIA Brev Deployment Instructions

### 1. Deploy with CLI
```bash
# Set environment variable
export NVIDIA_BREV_API_TOKEN="{self.api_token[:20]}..."

python -m deepracer_research.deployment.nvidia_brev.cli deploy \\
    --model-name {self.model_name} \\
    --track-arn "arn:aws:deepracer:us-east-1::track/reInvent2019_track" \\
    --reward-function reward_function.py \\
    --gpu-type {self.gpu_type.value} \\
    --deployment-mode {self.deployment_mode.value} \\
    --s3-bucket {s3_bucket} \\
    --auto-start
```

```python
from deepracer_research.deployment.nvidia_brev import NvidiaBrevDeploymentManager
from deepracer_research.deployment.nvidia_brev.config.deepracer_config import NvidiaBrevDeepRacerConfig

config = NvidiaBrevDeepRacerConfig.create_quick_training(
    model_name="{self.model_name}",
    track_arn="arn:aws:deepracer:us-east-1::track/reInvent2019_track",
    reward_function_code=open("reward_function.py").read(),
    api_token="{self.api_token[:20]}...",
    s3_bucket="{s3_bucket}",
    gpu_type=GPUType.{self.gpu_type.name},
    deployment_mode=DeploymentMode.{self.deployment_mode.name}
)

with NvidiaBrevDeploymentManager(config) as manager:
    result = manager.deploy()
    print(f"Access Jupyter: {{result['access_info']['jupyter']}}")
```

```bash
# Check status
python -m deepracer_research.deployment.nvidia_brev.cli status

# Create SSH tunnel for Jupyter
python -m deepracer_research.deployment.nvidia_brev.cli tunnel \\
    --instance-id your-instance-id \\
    --local-port 8888 \\
    --remote-port 8888
```

### 4. Access Training
- Training UI: http://localhost:8080 (via SSH tunnel)
- Jupyter Notebook: http://localhost:8888 (via SSH tunnel)
- TensorBoard: http://localhost:6006 (via SSH tunnel)

### 5. Cost Information
- GPU Type: {self.gpu_type.display_name}
- Deployment Mode: {self.deployment_mode.display_name}
- Estimated Cost: ${cost_estimate['total_cost']:.2f} for 8 hours
- Hourly Rate: ${cost_estimate['hourly_rate']:.2f}/hour

### 6. S3 Configuration
S3 Bucket: {s3_bucket}
Configuration files uploaded to: s3://{s3_bucket}/{self.model_name}/custom_files/

### 7. Cleanup
```bash
# Stop training and destroy instance
python -m deepracer_research.deployment.nvidia_brev.cli destroy \\
    --instance-id your-instance-id \\
    --force
```
"""

        for file_type, file_path in result["files"].items():
            instructions += f"- {file_type}: {file_path}\n"

        return instructions
