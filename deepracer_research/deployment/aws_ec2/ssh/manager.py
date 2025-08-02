from typing import List, Optional

from deepracer_research.deployment.aws_ec2.config.ssh_config import EC2SSHConfig
from deepracer_research.deployment.aws_ec2.ssh.connection import SSHConnectionManager
from deepracer_research.deployment.aws_ec2.ssh.execution import SSHCommandExecutor
from deepracer_research.deployment.aws_ec2.ssh.models import SSHCommand
from deepracer_research.deployment.aws_ec2.ssh.transfer import SSHFileTransfer


class EC2SSHManager:
    """Main SSH manager that orchestrates all SSH operations for EC2 instances"""

    def __init__(self, instance_id: str, hostname: str, ssh_config: Optional[EC2SSHConfig] = None):
        """Initialize SSH manager.

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

        self.connection = SSHConnectionManager(instance_id, hostname, ssh_config)
        self.executor = SSHCommandExecutor(instance_id, hostname, ssh_config)
        self.transfer = SSHFileTransfer(instance_id, hostname, ssh_config)

    @property
    def connected(self) -> bool:
        """Check if connected to the instance.

        Returns
        -------
        bool
            True if SSH connection is active.
        """
        return self.connection.connected

    def connect(self) -> bool:
        """Establish SSH connection to the EC2 instance.

        Returns
        -------
        bool
            True if connection was successful.
        """
        return self.connection.connect()

    def wait_for_instance_ready(self, timeout: int = 300, check_interval: int = 10) -> bool:
        """Wait for instance to be ready for SSH connections.

        Parameters
        ----------
        timeout : int, optional
            Maximum time to wait in seconds, by default 300.
        check_interval : int, optional
            Time between connection attempts in seconds, by default 10.

        Returns
        -------
        bool
            True if instance becomes ready, False if timeout.
        """
        return self.connection.wait_for_instance_ready(timeout, check_interval)

    def disconnect(self) -> None:
        """Disconnect from the instance."""
        self.connection.disconnect()

    def execute_command(self, command: str, timeout: Optional[int] = None, capture_output: bool = True) -> SSHCommand:
        """Execute command on remote instance.

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
        """
        return self.executor.execute_command(command, timeout, capture_output)

    def execute_script(self, script_content: str, script_name: str = "setup.sh") -> SSHCommand:
        """Execute a script on the remote instance.

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
        return self.executor.execute_script(script_content, script_name)

    def execute_sudo_command(self, command: str, timeout: Optional[int] = None) -> SSHCommand:
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
        return self.executor.execute_command_with_sudo(command, timeout)

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to remote instance.

        Parameters
        ----------
        local_path : str
            Local file path.
        remote_path : str
            Remote file path.

        Returns
        -------
        bool
            True if upload was successful.
        """
        return self.transfer.upload_file(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from remote instance.

        Parameters
        ----------
        remote_path : str
            Remote file path.
        local_path : str
            Local file path.

        Returns
        -------
        bool
            True if download was successful.
        """
        return self.transfer.download_file(remote_path, local_path)

    def upload_directory(self, local_dir: str, remote_dir: str, exclude_patterns: Optional[List[str]] = None) -> bool:
        """Upload directory to remote instance.

        Parameters
        ----------
        local_dir : str
            Local directory path.
        remote_dir : str
            Remote directory path.
        exclude_patterns : List[str], optional
            Patterns to exclude from upload, by default None.

        Returns
        -------
        bool
            True if upload was successful.
        """
        return self.transfer.upload_directory(local_dir, remote_dir, exclude_patterns)

    def sync_project(self, project_root: str, remote_project_dir: str = "~/deepracer-research") -> bool:
        """Sync entire DeepRacer research project to remote instance.

        Parameters
        ----------
        project_root : str
            Local project root directory.
        remote_project_dir : str, optional
            Remote project directory, by default "~/deepracer-research".

        Returns
        -------
        bool
            True if sync was successful.
        """
        return self.transfer.sync_project(project_root, remote_project_dir)

    def open_shell(self) -> None:
        """Open interactive SSH shell to the instance."""
        import subprocess

        ssh_args = self.ssh_config.get_ssh_command_args(self.hostname, key_name=None)

        try:
            subprocess.run(ssh_args, check=False)
        except KeyboardInterrupt:
            pass

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
        return self.executor.check_command_exists(command)
