import paramiko

from deepracer_research.deployment.nvidia_brev.config.ssh_config import SSHConfig
from deepracer_research.deployment.nvidia_brev.models import NvidiaBrevError


class SSHConnection:
    """Individual SSH connection wrapper

    Parameters
    ----------
    ssh_config : SSHConfig
        SSH configuration
    """

    def __init__(self, ssh_config: SSHConfig):
        self.ssh_config = ssh_config
        self.client = None

    def connect(self) -> paramiko.SSHClient:
        """Establish SSH connection.

        Returns
        -------
        paramiko.SSHClient
            Connected SSH client

        Raises
        ------
        NvidiaBrevError
            If connection fails
        """
        if self.client:
            return self.client

        try:
            self.client = paramiko.SSHClient()

            if self.ssh_config.known_hosts_policy == "ignore":
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            elif self.ssh_config.known_hosts_policy == "auto_add":
                self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            else:
                self.client.load_system_host_keys()
                self.client.set_missing_host_key_policy(paramiko.RejectPolicy())

            connect_kwargs = {
                "hostname": self.ssh_config.host,
                "port": self.ssh_config.port,
                "username": self.ssh_config.username,
                "timeout": self.ssh_config.connection_timeout,
                "compress": self.ssh_config.compression,
            }

            if self.ssh_config.private_key_content:
                from io import StringIO

                private_key = paramiko.RSAKey.from_private_key(StringIO(self.ssh_config.private_key_content))
                connect_kwargs["pkey"] = private_key
            elif self.ssh_config.private_key_path:
                connect_kwargs["key_filename"] = self.ssh_config.private_key_path
            elif self.ssh_config.use_ssh_agent:
                connect_kwargs["allow_agent"] = True

            self.client.connect(**connect_kwargs)
            return self.client

        except Exception as e:
            if self.client:
                self.client.close()
                self.client = None
            raise NvidiaBrevError(f"SSH connection failed: {str(e)}")

    def disconnect(self):
        """Close SSH connection."""
        if self.client:
            self.client.close()
            self.client = None

    def is_connected(self) -> bool:
        """Check if connection is active.

        Returns
        -------
        bool
            True if connected
        """
        if not self.client:
            return False

        try:
            transport = self.client.get_transport()
            return transport and transport.is_active()
        except:
            return False

    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
