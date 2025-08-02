from typing import Any, Dict

from deepracer_research.deployment.nvidia_brev.config.deepracer_config import NvidiaBrevDeepRacerConfig
from deepracer_research.deployment.nvidia_brev.ssh.ssh_manager import NvidiaBrevSSHManager


class TrainingManager:
    """Training management for DeepRacer on NVIDIA Brev instances"""

    def __init__(self, ssh_manager: NvidiaBrevSSHManager, config: NvidiaBrevDeepRacerConfig):
        self.ssh_manager = ssh_manager
        self.config = config

    def initialize(self) -> Dict[str, Any]:
        """Initialize training environment."""
        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "cd /home/ubuntu/deepracer-for-cloud && ./bin/init.sh", timeout=600
            )
            return {"success": exit_code == 0, "stdout": stdout, "stderr": stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_training(self) -> Dict[str, Any]:
        """Start training."""
        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "cd /home/ubuntu/deepracer-for-cloud && dr-start-training", timeout=120
            )
            return {"success": exit_code == 0, "stdout": stdout, "stderr": stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def stop_training(self) -> Dict[str, Any]:
        """Stop training."""
        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "cd /home/ubuntu/deepracer-for-cloud && dr-stop-training", timeout=60
            )
            return {"success": exit_code == 0, "stdout": stdout, "stderr": stderr}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get training status."""
        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "cd /home/ubuntu/deepracer-for-cloud && dr-logs -f 5", timeout=30
            )
            return {"success": exit_code == 0, "logs": stdout, "error": stderr if exit_code != 0 else None}
        except Exception as e:
            return {"success": False, "error": str(e)}
