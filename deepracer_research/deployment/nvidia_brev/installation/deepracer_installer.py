import time
from typing import Any, Dict, Optional

from deepracer_research.deployment.nvidia_brev.config.deepracer_config import NvidiaBrevDeepRacerConfig
from deepracer_research.deployment.nvidia_brev.ssh.ssh_executor import SSHCommand, SSHCommandExecutor
from deepracer_research.deployment.nvidia_brev.ssh.ssh_manager import NvidiaBrevSSHManager


class DeepRacerInstaller:
    """DeepRacer-for-Cloud installer for NVIDIA Brev instances

    Parameters
    ----------
    ssh_manager : NvidiaBrevSSHManager
        SSH manager for remote operations
    config : NvidiaBrevDeepRacerConfig
        DeepRacer configuration
    """

    def __init__(self, ssh_manager: NvidiaBrevSSHManager, config: NvidiaBrevDeepRacerConfig):
        self.ssh_manager = ssh_manager
        self.config = config
        self.executor = SSHCommandExecutor(ssh_manager)

    def install(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Install DeepRacer-for-Cloud on the instance.

        Parameters
        ----------
        progress_callback : Optional[callable], optional
            Callback function for progress updates, by default None

        Returns
        -------
        Dict[str, Any]
            Installation results
        """
        results = {}
        start_time = time.time()

        try:
            if progress_callback:
                progress_callback("Starting DeepRacer installation...")

            if progress_callback:
                progress_callback("Preparing system...")

            prep_result = self._prepare_system()
            results["system_preparation"] = prep_result

            if not prep_result["success"]:
                return self._create_error_result("System preparation failed", results)

            if progress_callback:
                progress_callback("Installing Docker...")

            docker_result = self._install_docker()
            results["docker_installation"] = docker_result

            if not docker_result["success"]:
                return self._create_error_result("Docker installation failed", results)

            if progress_callback:
                progress_callback("Installing DeepRacer-for-Cloud...")

            deepracer_result = self._install_deepracer_for_cloud()
            results["deepracer_installation"] = deepracer_result

            if not deepracer_result["success"]:
                return self._create_error_result("DeepRacer installation failed", results)

            if progress_callback:
                progress_callback("Configuring environment...")

            config_result = self._configure_environment()
            results["environment_configuration"] = config_result

            if not config_result["success"]:
                return self._create_error_result("Environment configuration failed", results)

            if progress_callback:
                progress_callback("Setting up reward function...")

            reward_result = self._setup_reward_function()
            results["reward_function_setup"] = reward_result

            if not reward_result["success"]:
                return self._create_error_result("Reward function setup failed", results)

            if progress_callback:
                progress_callback("Verifying installation...")

            verify_result = self._verify_installation()
            results["verification"] = verify_result

            duration = time.time() - start_time

            if progress_callback:
                progress_callback("Installation completed successfully!")

            return {
                "success": True,
                "message": "DeepRacer installation completed successfully",
                "duration": duration,
                "results": results,
            }

        except Exception as e:
            duration = time.time() - start_time
            return {
                "success": False,
                "message": f"Installation failed: {str(e)}",
                "duration": duration,
                "results": results,
                "error": str(e),
            }

    def _prepare_system(self) -> Dict[str, Any]:
        """Prepare the system for DeepRacer installation."""
        commands = [
            SSHCommand("sudo apt-get update -y", name="update_packages"),
            SSHCommand("sudo apt-get upgrade -y", name="upgrade_packages"),
            SSHCommand("sudo apt-get install -y curl wget git unzip awscli", name="install_basic_tools"),
            SSHCommand("curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -", name="add_docker_key"),
        ]

        command_results = self.executor.execute_commands(commands, stop_on_error=False)

        success = all(result.success for result in command_results)

        return {
            "success": success,
            "commands": len(commands),
            "results": [{"command": r.command.name, "success": r.success, "duration": r.duration} for r in command_results],
        }

    def _install_docker(self) -> Dict[str, Any]:
        """Install Docker and NVIDIA Docker support."""
        docker_results = self.executor.setup_docker()

        nvidia_results = self.executor.setup_nvidia_docker()

        all_results = docker_results + nvidia_results
        success = all(result.success for result in all_results)

        return {
            "success": success,
            "docker_commands": len(docker_results),
            "nvidia_commands": len(nvidia_results),
            "results": [{"command": r.command.name, "success": r.success, "duration": r.duration} for r in all_results],
        }

    def _install_deepracer_for_cloud(self) -> Dict[str, Any]:
        """Install DeepRacer-for-Cloud."""
        commands = [
            SSHCommand("cd /home/ubuntu", name="change_to_home"),
            SSHCommand("git clone https://github.com/aws-deepracer-community/deepracer-for-cloud.git", name="clone_repo"),
            SSHCommand("cd deepracer-for-cloud && chmod +x bin/*.sh", name="make_scripts_executable"),
        ]

        command_results = self.executor.execute_commands(commands, stop_on_error=True)

        success = all(result.success for result in command_results)

        return {
            "success": success,
            "commands": len(commands),
            "results": [{"command": r.command.name, "success": r.success, "duration": r.duration} for r in command_results],
        }

    def _configure_environment(self) -> Dict[str, Any]:
        """Configure the DeepRacer environment."""
        try:
            env_vars = self.config.get_environment_variables()

            env_content = []
            for key, value in env_vars.items():
                if key.startswith("DR_"):
                    env_content.append(f"{key}={value}")

            env_file_content = "\n".join(env_content)

            self.ssh_manager.upload_content(env_file_content, "/home/ubuntu/deepracer-for-cloud/.env")

            commands = [
                SSHCommand("mkdir -p ~/.aws", name="create_aws_dir"),
                SSHCommand(f"echo '[{self.config.aws_profile}]' > ~/.aws/config", name="create_aws_config"),
                SSHCommand("echo 'region = us-east-1' >> ~/.aws/config", name="set_aws_region"),
                SSHCommand("echo 'output = json' >> ~/.aws/config", name="set_aws_output"),
            ]

            command_results = self.executor.execute_commands(commands, stop_on_error=False)

            success = all(result.success for result in command_results)

            return {
                "success": success,
                "env_vars_count": len([k for k in env_vars.keys() if k.startswith("DR_")]),
                "commands": len(commands),
                "results": [{"command": r.command.name, "success": r.success} for r in command_results],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _setup_reward_function(self) -> Dict[str, Any]:
        """Setup the reward function."""
        try:
            self.ssh_manager.execute_command("mkdir -p /home/ubuntu/deepracer-for-cloud/custom_files")

            reward_function_path = "/home/ubuntu/deepracer-for-cloud/custom_files/reward_function.py"
            self.ssh_manager.upload_content(self.config.reward_function_code, reward_function_path)

            file_exists = self.ssh_manager.file_exists(reward_function_path)

            return {
                "success": file_exists,
                "reward_function_size": len(self.config.reward_function_code),
                "file_path": reward_function_path,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _verify_installation(self) -> Dict[str, Any]:
        """Verify the DeepRacer installation."""
        try:
            verification_results = {}

            exit_code, stdout, stderr = self.ssh_manager.execute_command("docker --version")
            verification_results["docker"] = exit_code == 0

            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi"
            )
            verification_results["nvidia_docker"] = exit_code == 0

            verification_results["deepracer_directory"] = self.ssh_manager.directory_exists("/home/ubuntu/deepracer-for-cloud")

            verification_results["env_file"] = self.ssh_manager.file_exists("/home/ubuntu/deepracer-for-cloud/.env")

            verification_results["reward_function"] = self.ssh_manager.file_exists(
                "/home/ubuntu/deepracer-for-cloud/custom_files/reward_function.py"
            )

            verification_results["init_script"] = self.ssh_manager.file_exists("/home/ubuntu/deepracer-for-cloud/bin/init.sh")

            all_checks_passed = all(verification_results.values())

            return {
                "success": all_checks_passed,
                "checks": verification_results,
                "passed": sum(verification_results.values()),
                "total": len(verification_results),
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _create_error_result(self, message: str, partial_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create an error result dictionary."""
        return {"success": False, "message": message, "partial_results": partial_results}

    def initialize_deepracer(self) -> Dict[str, Any]:
        """Initialize DeepRacer after installation."""
        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "cd /home/ubuntu/deepracer-for-cloud && ./bin/init.sh", timeout=600
            )

            return {"success": exit_code == 0, "exit_code": exit_code, "stdout": stdout, "stderr": stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def start_training(self) -> Dict[str, Any]:
        """Start DeepRacer training."""
        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "cd /home/ubuntu/deepracer-for-cloud && dr-start-training", timeout=120
            )

            return {"success": exit_code == 0, "exit_code": exit_code, "stdout": stdout, "stderr": stderr}

        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_training_status(self) -> Dict[str, Any]:
        """Get training status."""
        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_command(
                "cd /home/ubuntu/deepracer-for-cloud && dr-logs -f 5", timeout=30
            )

            return {"success": exit_code == 0, "logs": stdout, "error": stderr if exit_code != 0 else None}

        except Exception as e:
            return {"success": False, "error": str(e)}
