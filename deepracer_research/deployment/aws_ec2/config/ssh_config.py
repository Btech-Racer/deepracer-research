from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class EC2SSHConfig:
    """SSH configuration for EC2 instance connections

    Parameters
    ----------
    username : str, optional
        SSH username, by default "ubuntu".
    private_key_path : str, optional
        Path to private key file, by default None.
    ssh_port : int, optional
        SSH port number, by default 22.
    connection_timeout : int, optional
        SSH connection timeout in seconds, by default 30.
    retry_attempts : int, optional
        Number of connection retry attempts, by default 3.
    retry_delay : int, optional
        Delay between retry attempts in seconds, by default 5.
    strict_host_key_checking : bool, optional
        Enable strict host key checking, by default False.
    use_agent : bool, optional
        Use SSH agent for authentication, by default True.
    compression : bool, optional
        Enable SSH compression, by default True.
    keep_alive_interval : int, optional
        Keep-alive interval in seconds, by default 60.
    max_keep_alive_count : int, optional
        Maximum keep-alive count, by default 3.
    """

    username: str = "ubuntu"
    private_key_path: Optional[str] = None
    ssh_port: int = 22
    connection_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5
    strict_host_key_checking: bool = False
    use_agent: bool = True
    compression: bool = True
    keep_alive_interval: int = 60
    max_keep_alive_count: int = 3

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {
            "username": self.username,
            "private_key_path": self.private_key_path,
            "ssh_port": self.ssh_port,
            "connection_timeout": self.connection_timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            "strict_host_key_checking": self.strict_host_key_checking,
            "use_agent": self.use_agent,
            "compression": self.compression,
            "keep_alive_interval": self.keep_alive_interval,
            "max_keep_alive_count": self.max_keep_alive_count,
        }

    def get_ssh_options(self) -> Dict[str, Any]:
        """Get SSH client options for paramiko.

        Returns
        -------
        Dict[str, Any]
            Dictionary of SSH client options.
        """
        options = {
            "timeout": self.connection_timeout,
            "compress": self.compression,
        }

        return options

    def get_ssh_command_args(self, hostname: str, key_name: Optional[str] = None) -> list:
        """Generate SSH command arguments.

        Parameters
        ----------
        hostname : str
            Target hostname or IP address.
        key_name : str, optional
            EC2 key pair name (used if private_key_path not specified), by default None.

        Returns
        -------
        list
            List of SSH command arguments.
        """
        args = ["ssh"]

        if self.ssh_port != 22:
            args.extend(["-p", str(self.ssh_port)])

        if self.private_key_path:
            args.extend(["-i", self.private_key_path])
        elif key_name:
            key_path = f"~/.ssh/{key_name}.pem"
            args.extend(["-i", key_path])

        args.extend(["-o", f"ConnectTimeout={self.connection_timeout}"])
        args.extend(["-o", f"ServerAliveInterval={self.keep_alive_interval}"])
        args.extend(["-o", f"ServerAliveCountMax={self.max_keep_alive_count}"])

        if not self.strict_host_key_checking:
            args.extend(["-o", "StrictHostKeyChecking=no"])
            args.extend(["-o", "UserKnownHostsFile=/dev/null"])

        if self.compression:
            args.extend(["-o", "Compression=yes"])

        args.append(f"{self.username}@{hostname}")

        return args

    def validate(self) -> None:
        """Validate the SSH configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if not self.username:
            raise ValueError("Username cannot be empty")

        if self.ssh_port < 1 or self.ssh_port > 65535:
            raise ValueError("SSH port must be between 1 and 65535")

        if self.connection_timeout < 1:
            raise ValueError("Connection timeout must be positive")

        if self.retry_attempts < 0:
            raise ValueError("Retry attempts cannot be negative")

        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")

        if self.keep_alive_interval < 0:
            raise ValueError("Keep-alive interval cannot be negative")

        if self.max_keep_alive_count < 0:
            raise ValueError("Max keep-alive count cannot be negative")

        if self.private_key_path:
            key_path = Path(self.private_key_path).expanduser()
            if not key_path.exists():
                raise ValueError(f"Private key file not found: {self.private_key_path}")

            if key_path.stat().st_mode & 0o077:
                raise ValueError(f"Private key file has too open permissions: {self.private_key_path}")

    @classmethod
    def for_ec2_default(cls, private_key_path: Optional[str] = None) -> "EC2SSHConfig":
        """Create default SSH configuration for EC2 instances.

        Parameters
        ----------
        private_key_path : str, optional
            Path to private key file, by default None.

        Returns
        -------
        EC2SSHConfig
            Default SSH configuration for EC2.
        """
        return cls(
            username="ubuntu",
            private_key_path=private_key_path,
            strict_host_key_checking=False,
            connection_timeout=30,
            retry_attempts=5,
            retry_delay=10,
        )

    @classmethod
    def for_amazon_linux(cls, private_key_path: Optional[str] = None) -> "EC2SSHConfig":
        """Create SSH configuration for Amazon Linux instances.

        Parameters
        ----------
        private_key_path : str, optional
            Path to private key file, by default None.

        Returns
        -------
        EC2SSHConfig
            SSH configuration for Amazon Linux.
        """
        return cls(
            username="ec2-user",
            private_key_path=private_key_path,
            strict_host_key_checking=False,
            connection_timeout=30,
            retry_attempts=5,
            retry_delay=10,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EC2SSHConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration parameters.

        Returns
        -------
        EC2SSHConfig
            Configuration instance created from dictionary.
        """
        return cls(
            username=data.get("username", "ubuntu"),
            private_key_path=data.get("private_key_path"),
            ssh_port=data.get("ssh_port", 22),
            connection_timeout=data.get("connection_timeout", 30),
            retry_attempts=data.get("retry_attempts", 3),
            retry_delay=data.get("retry_delay", 5),
            strict_host_key_checking=data.get("strict_host_key_checking", False),
            use_agent=data.get("use_agent", True),
            compression=data.get("compression", True),
            keep_alive_interval=data.get("keep_alive_interval", 60),
            max_keep_alive_count=data.get("max_keep_alive_count", 3),
        )
