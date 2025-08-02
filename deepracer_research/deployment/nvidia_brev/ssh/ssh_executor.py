import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from deepracer_research.deployment.nvidia_brev.ssh.ssh_manager import NvidiaBrevSSHManager


@dataclass
class SSHCommand:
    """SSH command definition

    Parameters
    ----------
    command : str
        Command to execute
    name : str, optional
        Command name for identification, by default None
    timeout : int, optional
        Command timeout in seconds, by default None
    environment : Dict[str, str], optional
        Environment variables, by default None
    get_pty : bool, optional
        Whether to allocate PTY, by default False
    """

    command: str
    name: Optional[str] = None
    timeout: Optional[int] = None
    environment: Optional[Dict[str, str]] = None
    get_pty: bool = False


@dataclass
class SSHCommandResult:
    """SSH command execution result

    Parameters
    ----------
    command : SSHCommand
        Original command
    exit_code : int
        Command exit code
    stdout : str
        Standard output
    stderr : str
        Standard error
    duration : float
        Execution duration in seconds
    success : bool
        Whether command succeeded (exit_code == 0)
    """

    command: SSHCommand
    exit_code: int
    stdout: str
    stderr: str
    duration: float

    @property
    def success(self) -> bool:
        """Check if command succeeded."""
        return self.exit_code == 0


class SSHCommandExecutor:
    """Batch SSH command executor

    Executes multiple SSH commands in sequence with error handling
    and result collection.

    Parameters
    ----------
    ssh_manager : NvidiaBrevSSHManager
        SSH manager instance
    """

    def __init__(self, ssh_manager: NvidiaBrevSSHManager):
        self.ssh_manager = ssh_manager

    def execute_commands(
        self, commands: List[Union[str, SSHCommand]], stop_on_error: bool = True, parallel: bool = False
    ) -> List[SSHCommandResult]:
        """Execute a list of commands.

        Parameters
        ----------
        commands : List[Union[str, SSHCommand]]
            List of commands to execute
        stop_on_error : bool, optional
            Whether to stop on first error, by default True
        parallel : bool, optional
            Whether to execute commands in parallel, by default False

        Returns
        -------
        List[SSHCommandResult]
            List of command results
        """
        ssh_commands = []
        for i, cmd in enumerate(commands):
            if isinstance(cmd, str):
                ssh_commands.append(SSHCommand(command=cmd, name=f"command_{i}"))
            else:
                ssh_commands.append(cmd)

        if parallel:
            return self._execute_parallel(ssh_commands, stop_on_error)
        else:
            return self._execute_sequential(ssh_commands, stop_on_error)

    def _execute_sequential(self, commands: List[SSHCommand], stop_on_error: bool) -> List[SSHCommandResult]:
        """Execute commands sequentially.

        Parameters
        ----------
        commands : List[SSHCommand]
            Commands to execute
        stop_on_error : bool
            Whether to stop on error

        Returns
        -------
        List[SSHCommandResult]
            Command results
        """
        results = []

        for command in commands:
            start_time = time.time()

            try:
                exit_code, stdout, stderr = self.ssh_manager.execute_command(
                    command.command, timeout=command.timeout, get_pty=command.get_pty, environment=command.environment
                )

                duration = time.time() - start_time

                result = SSHCommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr=stderr, duration=duration)

                results.append(result)

                if not result.success and stop_on_error:
                    break

            except Exception as e:
                duration = time.time() - start_time

                result = SSHCommandResult(command=command, exit_code=-1, stdout="", stderr=str(e), duration=duration)

                results.append(result)

                if stop_on_error:
                    break

        return results

    def _execute_parallel(self, commands: List[SSHCommand], stop_on_error: bool) -> List[SSHCommandResult]:
        """Execute commands in parallel.

        Note: This is a simplified implementation. True parallelism
        would require multiple SSH connections.

        Parameters
        ----------
        commands : List[SSHCommand]
            Commands to execute
        stop_on_error : bool
            Whether to stop on error

        Returns
        -------
        List[SSHCommandResult]
            Command results
        """
        return self._execute_sequential(commands, stop_on_error)

    def execute_setup_script(self, script_lines: List[str], name: str = "setup_script") -> SSHCommandResult:
        """Execute a setup script from lines.

        Parameters
        ----------
        script_lines : List[str]
            Script lines
        name : str, optional
            Script name, by default "setup_script"

        Returns
        -------
        SSHCommandResult
            Execution result
        """
        script_content = "\n".join(script_lines)

        start_time = time.time()

        try:
            exit_code, stdout, stderr = self.ssh_manager.execute_script(script_content, script_name=f"{name}.sh")

            duration = time.time() - start_time

            command = SSHCommand(command=f"execute_script:{name}")

            return SSHCommandResult(command=command, exit_code=exit_code, stdout=stdout, stderr=stderr, duration=duration)

        except Exception as e:
            duration = time.time() - start_time

            command = SSHCommand(command=f"execute_script:{name}")

            return SSHCommandResult(command=command, exit_code=-1, stdout="", stderr=str(e), duration=duration)

    def install_packages(self, packages: List[str], package_manager: str = "apt") -> SSHCommandResult:
        """Install packages using system package manager.

        Parameters
        ----------
        packages : List[str]
            Packages to install
        package_manager : str, optional
            Package manager to use, by default "apt"

        Returns
        -------
        SSHCommandResult
            Installation result
        """
        if package_manager == "apt":
            commands = ["sudo apt-get update -y", f"sudo apt-get install -y {' '.join(packages)}"]
        elif package_manager == "yum":
            commands = [f"sudo yum install -y {' '.join(packages)}"]
        elif package_manager == "pip":
            commands = [f"pip install {' '.join(packages)}"]
        else:
            raise ValueError(f"Unsupported package manager: {package_manager}")

        results = self.execute_commands(commands, stop_on_error=True)

        total_duration = sum(r.duration for r in results)
        success = all(r.success for r in results)

        combined_stdout = "\n".join(r.stdout for r in results)
        combined_stderr = "\n".join(r.stderr for r in results)

        command = SSHCommand(command=f"install_packages:{' '.join(packages)}")

        return SSHCommandResult(
            command=command,
            exit_code=0 if success else 1,
            stdout=combined_stdout,
            stderr=combined_stderr,
            duration=total_duration,
        )

    def setup_docker(self) -> List[SSHCommandResult]:
        """Set up Docker on the instance.

        Returns
        -------
        List[SSHCommandResult]
            Setup results
        """
        commands = [
            SSHCommand("sudo apt-get update -y", name="update_packages"),
            SSHCommand("sudo apt-get install -y docker.io docker-compose", name="install_docker"),
            SSHCommand("sudo usermod -aG docker $USER", name="add_user_to_docker"),
            SSHCommand("sudo systemctl start docker", name="start_docker"),
            SSHCommand("sudo systemctl enable docker", name="enable_docker"),
        ]

        return self.execute_commands(commands, stop_on_error=False)

    def setup_nvidia_docker(self) -> List[SSHCommandResult]:
        """Set up NVIDIA Docker support.

        Returns
        -------
        List[SSHCommandResult]
            Setup results
        """
        commands = [
            SSHCommand("distribution=$(. /etc/os-release;echo $ID$VERSION_ID)", name="get_distribution"),
            SSHCommand("curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -", name="add_nvidia_key"),
            SSHCommand(
                "curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list",
                name="add_nvidia_repo",
            ),
            SSHCommand("sudo apt-get update", name="update_packages"),
            SSHCommand("sudo apt-get install -y nvidia-docker2", name="install_nvidia_docker"),
            SSHCommand("sudo systemctl restart docker", name="restart_docker"),
        ]

        return self.execute_commands(commands, stop_on_error=False)

    def check_gpu_availability(self) -> SSHCommandResult:
        """Check GPU availability and status.

        Returns
        -------
        SSHCommandResult
            GPU check result
        """
        command = SSHCommand(
            "nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,memory.free --format=csv", name="check_gpu"
        )

        return self.execute_commands([command])[0]
