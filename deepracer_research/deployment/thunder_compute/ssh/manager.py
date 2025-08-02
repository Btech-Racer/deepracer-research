from typing import Optional

from deepracer_research.deployment.thunder_compute.config.ssh_config import SSHConfig
from deepracer_research.deployment.thunder_compute.ssh.connection import SSHConnectionManager
from deepracer_research.deployment.thunder_compute.ssh.execution import SSHCommandExecutor
from deepracer_research.deployment.thunder_compute.ssh.models.ssh_command import SSHCommand
from deepracer_research.deployment.thunder_compute.ssh.transfer import SSHFileTransfer


class SSHManager:
    """Main SSH manager that orchestrates all SSH operations for Thunder Compute instances"""

    def __init__(self, instance_uuid: str, ssh_config: Optional[SSHConfig] = None, thunder_cli_index: Optional[str] = None):
        """Initialize SSH manager.

        Parameters
        ----------
        instance_uuid : str
            Thunder Compute instance UUID.
        ssh_config : SSHConfig, optional
            SSH configuration, by default None.
        thunder_cli_index : str, optional
            Thunder CLI index for this instance, by default None.
        """
        self.instance_uuid = instance_uuid
        self.thunder_cli_index = thunder_cli_index
        self.ssh_config = ssh_config or SSHConfig()

        self.connection = SSHConnectionManager(instance_uuid, ssh_config, thunder_cli_index)
        self.executor = SSHCommandExecutor(instance_uuid, ssh_config, thunder_cli_index)
        self.transfer = SSHFileTransfer(instance_uuid, ssh_config, thunder_cli_index)

    @property
    def connected(self) -> bool:
        """Check if connected to the instance."""
        return self.connection.connected

    def setup_tnr_connection(self, skip_tnr: bool = False) -> bool:
        """Setup Thunder Compute CLI connection.

        Parameters
        ----------
        skip_tnr : bool, optional
            Skip TNR CLI setup and use direct SSH instead, by default False.
        """
        return self.connection.setup_tnr_connection(skip_tnr)

    def wait_for_instance_ready(self, timeout: int = 300, check_interval: int = 10) -> bool:
        """Wait for instance to be ready for SSH connections."""
        return self.connection.wait_for_instance_ready(timeout, check_interval)

    def disconnect(self) -> None:
        """Disconnect from the instance."""
        self.connection.disconnect()

    def execute_command(self, command: str, timeout: Optional[int] = None, capture_output: bool = True) -> SSHCommand:
        """Execute command on remote instance."""
        return self.executor.execute_command(command, timeout, capture_output)

    def execute_script(self, script_content: str, script_name: str = "setup.sh") -> SSHCommand:
        """Execute a script on the remote instance."""
        return self.executor.execute_script(script_content, script_name)

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to remote instance."""
        return self.transfer.upload_file(local_path, remote_path)

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from remote instance."""
        return self.transfer.download_file(remote_path, local_path)

    def upload_directory(self, local_dir: str, remote_dir: str, exclude_patterns: Optional[list] = None) -> bool:
        """Upload directory to remote instance."""
        return self.transfer.upload_directory(local_dir, remote_dir, exclude_patterns)

    def sync_project(self, project_root: str, remote_project_dir: str = "~/deepracer-research") -> bool:
        """Sync entire DeepRacer research project to remote instance."""
        return self.transfer.sync_project(project_root, remote_project_dir)

    def open_shell(self) -> None:
        """Open interactive SSH shell to the instance."""
        if self.ssh_config.use_tnr_cli:
            import subprocess

            subprocess.run(["ssh", self.transfer.tnr_alias], check=False)
        else:
            import subprocess

            ssh_command = ["ssh", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no"]

            if self.ssh_config.ssh_key_path:
                ssh_command.extend(["-i", self.ssh_config.ssh_key_path])

            ssh_command.append(f"{self.ssh_config.username}@{self.instance_uuid}")
            subprocess.run(ssh_command, check=False)
