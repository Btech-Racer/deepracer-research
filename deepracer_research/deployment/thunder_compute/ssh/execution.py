import os
import subprocess
import tempfile
from typing import Optional

from deepracer_research.deployment.thunder_compute.config.ssh_config import SSHConfig
from deepracer_research.deployment.thunder_compute.ssh.models.ssh_command import SSHCommand
from deepracer_research.deployment.thunder_compute.ssh.models.ssh_connection_error import SSHConnectionError
from deepracer_research.utils.logger import debug


class SSHCommandExecutor:
    """Executor for SSH commands on Thunder Compute instances"""

    def __init__(self, instance_uuid: str, ssh_config: Optional[SSHConfig] = None, thunder_cli_index: Optional[str] = None):
        """Initialize SSH command executor.

        Parameters
        ----------
        instance_uuid : str
            Thunder Compute instance UUID.
        ssh_config : SSHConfig, optional
            SSH configuration, by default None (uses defaults).
        thunder_cli_index : str, optional
            Thunder CLI index for this instance, by default None.
        """
        self.instance_uuid = instance_uuid
        self.ssh_config = ssh_config or SSHConfig()
        self.thunder_cli_index = thunder_cli_index
        self.tnr_alias = f"tnr-{instance_uuid[:8]}"

    def execute_command(self, command: str, timeout: Optional[int] = None, capture_output: bool = True) -> SSHCommand:
        """Execute command on remote instance.

        Parameters
        ------------
        command : str
            Command to execute.
        timeout : int, optional
            Command timeout in seconds, by default None.
        capture_output : bool, optional
            Whether to capture command output, by default True.

        Returns
        -------
        SSHCommand
            SSH command result.

        Raises
        ------
        SSHConnectionError
            If SSH execution fails.
        """
        if self.ssh_config.use_tnr_cli:
            ssh_command = ["ssh", self.tnr_alias, command]
        else:
            if hasattr(self, "thunder_cli_index") and self.thunder_cli_index is not None:
                ssh_command = ["ssh", f"tnr-{self.thunder_cli_index}", command]
            else:
                ssh_command = [
                    "ssh",
                    "-o",
                    "ConnectTimeout=30",
                    "-o",
                    "StrictHostKeyChecking=no",
                    f"{self.ssh_config.username}@{self.instance_uuid}",
                    command,
                ]

                if self.ssh_config.ssh_key_path:
                    ssh_command.extend(["-i", self.ssh_config.ssh_key_path])

        try:
            debug("Executing SSH command", extra={"command": " ".join(ssh_command)})
            result = subprocess.run(
                ssh_command, capture_output=capture_output, text=True, timeout=timeout or self.ssh_config.connect_timeout
            )

            return SSHCommand(
                command=command,
                stdout=result.stdout,
                stderr=result.stderr,
                return_code=result.returncode,
                success=result.returncode == 0,
            )

        except subprocess.TimeoutExpired:
            raise SSHConnectionError(f"SSH command timed out: {command}")
        except Exception as e:
            raise SSHConnectionError(f"SSH command failed: {e}")

    def execute_script(self, script_content: str, script_name: str = "setup.sh") -> SSHCommand:
        """Execute a script on the remote instance.

        Parameters
        ------------
        script_content : str
            Content of the script to execute.
        script_name : str, optional
            Name for the script file, by default "setup.sh".

        Returns
        -------
        SSHCommand
            SSH command result.

        Raises
        ------
        SSHConnectionError
            If script execution fails.
        """
        from deepracer_research.deployment.thunder_compute.ssh.transfer import SSHFileTransfer

        remote_script_path = f"/tmp/{script_name}"
        transfer = SSHFileTransfer(self.instance_uuid, self.ssh_config)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            local_script_path = f.name

        try:
            transfer.upload_file(local_script_path, remote_script_path)

            chmod_result = self.execute_command(f"chmod +x {remote_script_path}")
            if not chmod_result.success:
                raise SSHConnectionError(f"Failed to make script executable: {chmod_result.stderr}")

            return self.execute_command(f"bash {remote_script_path}")

        finally:
            os.unlink(local_script_path)

            try:
                self.execute_command(f"rm -f {remote_script_path}")
            except:
                pass
