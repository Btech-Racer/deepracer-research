import time
from typing import Optional

import paramiko

from deepracer_research.deployment.aws_ec2.config.ssh_config import EC2SSHConfig
from deepracer_research.deployment.aws_ec2.ssh.exceptions import SSHConnectionError
from deepracer_research.utils.logger import debug, error, info, warning


class SSHConnectionManager:
    """Manages SSH connections to EC2 instances"""

    def __init__(self, instance_id: str, hostname: str, ssh_config: Optional[EC2SSHConfig] = None):
        """Initialize SSH connection manager.

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
        self.client: Optional[paramiko.SSHClient] = None
        self._connected = False

    @property
    def connected(self) -> bool:
        """Check if connected to the instance.

        Returns
        -------
        bool
            True if SSH connection is active.
        """
        return self._connected and self.client is not None

    def connect(self) -> bool:
        """Establish SSH connection to the EC2 instance.

        Returns
        -------
        bool
            True if connection was successful.

        Raises
        ------
        SSHConnectionError
            If connection fails.
        """
        if self.connected:
            return True

        try:
            info(
                "Establishing SSH connection",
                extra={"instance_id": self.instance_id, "hostname": self.hostname, "username": self.ssh_config.username},
            )

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

            connect_kwargs = {
                "hostname": self.hostname,
                "username": self.ssh_config.username,
                "port": self.ssh_config.ssh_port,
                "timeout": self.ssh_config.connection_timeout,
                "compress": self.ssh_config.compression,
            }

            if self.ssh_config.private_key_path:
                connect_kwargs["key_filename"] = self.ssh_config.private_key_path
            elif self.ssh_config.use_agent:
                connect_kwargs["look_for_keys"] = True

            for attempt in range(self.ssh_config.retry_attempts + 1):
                try:
                    self.client.connect(**connect_kwargs)
                    self._connected = True

                    info(
                        "SSH connection established",
                        extra={"instance_id": self.instance_id, "hostname": self.hostname, "attempt": attempt + 1},
                    )

                    return True

                except (paramiko.AuthenticationException, paramiko.SSHException, ConnectionError) as e:
                    if attempt < self.ssh_config.retry_attempts:
                        warning(
                            "SSH connection failed, retrying",
                            extra={
                                "instance_id": self.instance_id,
                                "hostname": self.hostname,
                                "attempt": attempt + 1,
                                "error": str(e),
                            },
                        )
                        time.sleep(self.ssh_config.retry_delay)
                    else:
                        raise e

            return False

        except Exception as e:
            error("SSH connection failed", extra={"instance_id": self.instance_id, "hostname": self.hostname, "error": str(e)})

            self._cleanup_connection()

            raise SSHConnectionError(
                f"Failed to connect to EC2 instance: {e}",
                instance_id=self.instance_id,
                hostname=self.hostname,
                port=self.ssh_config.ssh_port,
            )

    def disconnect(self) -> None:
        """Disconnect from the EC2 instance."""
        if self.connected:
            info("Disconnecting SSH connection", extra={"instance_id": self.instance_id, "hostname": self.hostname})

            self._cleanup_connection()

    def wait_for_instance_ready(self, timeout: int = 300, check_interval: int = 10) -> bool:
        """Wait for EC2 instance to be ready for SSH connections.

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
        info(
            "Waiting for EC2 instance to be ready",
            extra={"instance_id": self.instance_id, "hostname": self.hostname, "timeout": timeout},
        )

        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                if self.connect():
                    info("EC2 instance is ready for SSH", extra={"instance_id": self.instance_id, "hostname": self.hostname})
                    return True

            except SSHConnectionError:
                debug(
                    "Instance not yet ready for SSH",
                    extra={"instance_id": self.instance_id, "hostname": self.hostname, "elapsed": time.time() - start_time},
                )

            time.sleep(check_interval)

        warning(
            "Timeout waiting for EC2 instance to be ready",
            extra={"instance_id": self.instance_id, "hostname": self.hostname, "timeout": timeout},
        )

        return False

    def _cleanup_connection(self) -> None:
        """Clean up SSH connection resources."""
        if self.client:
            try:
                self.client.close()
            except Exception:
                pass
            finally:
                self.client = None
                self._connected = False
