import time
from typing import Optional

from deepracer_research.deployment.aws_ec2.config.ssh_config import EC2SSHConfig
from deepracer_research.deployment.aws_ec2.ssh.connection import SSHConnectionManager
from deepracer_research.deployment.aws_ec2.ssh.exceptions import SSHConnectionError
from deepracer_research.deployment.aws_ec2.ssh.models import SSHCommand
from deepracer_research.utils import debug, error, info


class SSHCommandExecutor:
    """Executes commands on EC2 instances via SSH"""

    def __init__(self, instance_id: str, hostname: str, ssh_config: Optional[EC2SSHConfig] = None):
        """Initialize SSH command executor.

        Parameters
        ----------
        instance_id : str
            EC2 instance ID.
        hostname : str
            Instance hostname or IP address.
        ssh_config : EC2SSHConfig, optional
            SSH configuration, by default None.
        """
        self.instance_id = instance_id
        self.hostname = hostname
        self.ssh_config = ssh_config or EC2SSHConfig.for_ec2_default()
        self.connection_manager = SSHConnectionManager(instance_id, hostname, ssh_config)

    def execute_command(self, command: str, timeout: Optional[int] = None, capture_output: bool = True) -> SSHCommand:
        """Execute a command on the EC2 instance.

        Parameters
        ----------
        command : str
            Command to execute.
        timeout : int, optional
            Command timeout in seconds, by default None.
        capture_output : bool, optional
            Whether to capture command output, by default True.

        Returns
        -------
        SSHCommand
            Command execution result.

        Raises
        ------
        SSHConnectionError
            If SSH connection fails.
        """
        if not self.connection_manager.connect():
            raise SSHConnectionError(
                "Cannot execute command: SSH connection failed", instance_id=self.instance_id, hostname=self.hostname
            )

        debug("Executing SSH command", extra={"instance_id": self.instance_id, "command": command, "timeout": timeout})

        start_time = time.time()
        ssh_command = SSHCommand(command=command)

        try:
            stdin, stdout, stderr = self.connection_manager.client.exec_command(command, timeout=timeout)

            if capture_output:
                ssh_command.stdout = stdout.read().decode("utf-8")
                ssh_command.stderr = stderr.read().decode("utf-8")

            ssh_command.exit_code = stdout.channel.recv_exit_status()
            ssh_command.execution_time = time.time() - start_time

            debug(
                "SSH command completed",
                extra={
                    "instance_id": self.instance_id,
                    "command": command,
                    "exit_code": ssh_command.exit_code,
                    "execution_time": ssh_command.execution_time,
                },
            )

        except Exception as e:
            ssh_command.execution_time = time.time() - start_time
            ssh_command.timed_out = "timeout" in str(e).lower()
            ssh_command.stderr = str(e)

            error("SSH command execution failed", extra={"instance_id": self.instance_id, "command": command, "error": str(e)})

        return ssh_command

    def execute_script(self, script_content: str, script_name: str = "setup.sh") -> SSHCommand:
        """Execute a script on the EC2 instance.

        Parameters
        ----------
        script_content : str
            Content of the script to execute.
        script_name : str, optional
            Name for the script file, by default "setup.sh".

        Returns
        -------
        SSHCommand
            Script execution result.
        """
        info(
            "Executing script on EC2 instance",
            extra={"instance_id": self.instance_id, "script_name": script_name, "script_size": len(script_content)},
        )

        remote_script_path = f"/tmp/{script_name}"

        upload_command = f"cat > {remote_script_path} << 'EOF'\n{script_content}\nEOF"
        upload_result = self.execute_command(upload_command)

        if not upload_result.success:
            error("Failed to upload script", extra={"instance_id": self.instance_id, "script_name": script_name})
            return upload_result

        make_executable = self.execute_command(f"chmod +x {remote_script_path}")
        if not make_executable.success:
            return make_executable

        result = self.execute_command(f"bash {remote_script_path}")

        self.execute_command(f"rm -f {remote_script_path}", capture_output=False)

        return result

    def execute_command_with_sudo(self, command: str, timeout: Optional[int] = None) -> SSHCommand:
        """Execute a command with sudo privileges.

        Parameters
        ----------
        command : str
            Command to execute with sudo.
        timeout : int, optional
            Command timeout in seconds, by default None.

        Returns
        -------
        SSHCommand
            Command execution result.
        """
        sudo_command = f"sudo {command}"
        return self.execute_command(sudo_command, timeout)

    def check_command_exists(self, command: str) -> bool:
        """Check if a command exists on the remote system.

        Parameters
        ----------
        command : str
            Command to check for existence.

        Returns
        -------
        bool
            True if command exists.
        """
        result = self.execute_command(f"which {command}")
        return result.success
