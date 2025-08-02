import subprocess
import time
from typing import Optional

from deepracer_research.deployment.thunder_compute.config.ssh_config import SSHConfig
from deepracer_research.deployment.thunder_compute.ssh.models.ssh_connection_error import SSHConnectionError
from deepracer_research.utils import debug, info, warning


class SSHConnectionManager:
    """Manager for SSH connections to Thunder Compute instances"""

    def __init__(self, instance_uuid: str, ssh_config: Optional[SSHConfig] = None, thunder_cli_index: Optional[str] = None):
        """Initialize SSH connection manager.

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
        self.connected = False
        self.tnr_alias = f"tnr-{instance_uuid[:8]}"

    def setup_tnr_connection(self, skip_tnr: bool = False) -> bool:
        """Setup Thunder Compute CLI connection.

        Parameters
        ----------
        skip_tnr : bool, optional
            Skip TNR CLI setup and use direct SSH instead, by default False.

        Returns
        -------
        bool
            True if setup successful.

        Raises
        ------
        SSHConnectionError
            If Thunder Compute CLI setup fails and direct SSH is not available.
        """
        if skip_tnr:
            info("Skipping Thunder Compute CLI connection, using direct SSH")
            self.ssh_config.use_tnr_cli = False
            self.connected = True
            return True

        info("Setting up Thunder Compute CLI connection", extra={"instance_uuid": self.instance_uuid})

        try:
            result = subprocess.run(["tnr", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                warning("Thunder Compute CLI (tnr) not found. Falling back to direct SSH.")
                self.ssh_config.use_tnr_cli = False
                self.connected = True
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            warning("Thunder Compute CLI (tnr) not found. Falling back to direct SSH.")
            self.ssh_config.use_tnr_cli = False
            self.connected = True
            return True

        try:
            connection_id = self.thunder_cli_index if self.thunder_cli_index is not None else self.instance_uuid
            info(f"Connecting via Thunder Compute CLI using {'index' if self.thunder_cli_index else 'UUID'}: {connection_id}")

            process = subprocess.Popen(
                ["tnr", "connect", connection_id],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            try:
                stdout, stderr = process.communicate(timeout=30)
                returncode = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                warning("Thunder Compute CLI connection timed out. Falling back to direct SSH.")
                self.ssh_config.use_tnr_cli = False
                self.connected = True
                return True

            if returncode == 0:
                self.connected = True
                info("Successfully connected via Thunder Compute CLI")
                return True
            else:
                warning("Thunder CLI connection failed. Falling back to direct SSH.", extra={"stderr": stderr})
                self.ssh_config.use_tnr_cli = False
                self.connected = True
                return True

        except Exception as e:
            warning(f"Thunder CLI connection failed: {e}. Falling back to direct SSH.")
            self.ssh_config.use_tnr_cli = False
            self.connected = True
            return True

    def wait_for_instance_ready(self, timeout: int = 300, check_interval: int = 10) -> bool:
        """Wait for instance to be ready for SSH connections.

        Parameters
        ------------
        timeout : int, optional
            Maximum time to wait in seconds, by default 300.
        check_interval : int, optional
            Time between checks in seconds, by default 10.

        Returns
        -------
        bool
            True if instance is ready.

        Raises
        ------
        SSHConnectionError
            If instance doesn't become ready within timeout.
        """
        info("Waiting for instance to be ready", extra={"instance_uuid": self.instance_uuid, "timeout": timeout})

        from deepracer_research.deployment.thunder_compute.ssh.execution import SSHCommandExecutor

        executor = SSHCommandExecutor(self.instance_uuid, self.ssh_config, self.thunder_cli_index)

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                result = executor.execute_command("echo 'ready'", timeout=5)
                if result.success:
                    info("Instance is ready for SSH connections")
                    return True
            except Exception as e:
                debug("Instance not ready yet", extra={"error": str(e)})

            time.sleep(check_interval)

        raise SSHConnectionError(f"Instance {self.instance_uuid} not ready within {timeout} seconds")

    def disconnect(self) -> None:
        """Disconnect from the instance."""
        info("Disconnecting from instance", extra={"instance_uuid": self.instance_uuid})
        self.connected = False
