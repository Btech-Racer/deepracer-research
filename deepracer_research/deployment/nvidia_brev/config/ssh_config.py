from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SSHConfig:
    """SSH configuration for NVIDIA Brev instances

    Parameters
    ----------
    username : str, optional
        SSH username, by default "ubuntu".
    private_key_path : str, optional
        Path to SSH private key file, by default None (use agent).
    private_key_content : str, optional
        SSH private key content as string, by default None.
    host : str, optional
        SSH host/IP address, by default None (auto-detected).
    port : int, optional
        SSH port number, by default 22.
    connection_timeout : int, optional
        SSH connection timeout in seconds, by default 30.
    command_timeout : int, optional
        Command execution timeout in seconds, by default 300.
    max_retries : int, optional
        Maximum connection retry attempts, by default 3.
    retry_delay : int, optional
        Delay between retry attempts in seconds, by default 5.
    use_ssh_agent : bool, optional
        Whether to use SSH agent for authentication, by default True.
    forward_agent : bool, optional
        Whether to enable SSH agent forwarding, by default False.
    compression : bool, optional
        Whether to enable SSH compression, by default True.
    keep_alive_interval : int, optional
        SSH keep-alive interval in seconds, by default 60.
    known_hosts_policy : str, optional
        Known hosts policy ('strict', 'auto_add', 'ignore'), by default 'auto_add'.
    additional_options : Dict[str, Any], optional
        Additional SSH client options, by default empty dict.
    """

    username: str = "ubuntu"
    private_key_path: Optional[str] = None
    private_key_content: Optional[str] = None
    host: Optional[str] = None
    port: int = 22

    connection_timeout: int = 30
    command_timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 5

    use_ssh_agent: bool = True
    forward_agent: bool = False
    compression: bool = True
    keep_alive_interval: int = 60
    known_hosts_policy: str = "auto_add"

    additional_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Post-initialization validation."""
        self.validate()

    def validate(self) -> None:
        """Validate the SSH configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if not self.username:
            raise ValueError("Username is required")

        if self.port <= 0 or self.port > 65535:
            raise ValueError("Port must be between 1 and 65535")

        if self.connection_timeout <= 0:
            raise ValueError("Connection timeout must be positive")

        if self.command_timeout <= 0:
            raise ValueError("Command timeout must be positive")

        if self.max_retries < 0:
            raise ValueError("Max retries cannot be negative")

        if self.retry_delay < 0:
            raise ValueError("Retry delay cannot be negative")

        if self.keep_alive_interval < 0:
            raise ValueError("Keep alive interval cannot be negative")

        if self.known_hosts_policy not in {"strict", "auto_add", "ignore"}:
            raise ValueError("Known hosts policy must be 'strict', 'auto_add', or 'ignore'")

        if self.private_key_path and not Path(self.private_key_path).exists():
            raise ValueError(f"Private key file not found: {self.private_key_path}")

    @property
    def has_private_key(self) -> bool:
        """Check if private key is configured.

        Returns
        -------
        bool
            True if private key path or content is provided
        """
        return bool(self.private_key_path or self.private_key_content)

    @property
    def ssh_command_args(self) -> List[str]:
        """Get SSH command arguments.

        Returns
        -------
        List[str]
            SSH command arguments
        """
        args = [
            "-o",
            f"ConnectTimeout={self.connection_timeout}",
            "-o",
            f"ServerAliveInterval={self.keep_alive_interval}",
            "-o",
            "ServerAliveCountMax=3",
            "-p",
            str(self.port),
        ]

        if self.compression:
            args.extend(["-o", "Compression=yes"])

        if self.forward_agent:
            args.extend(["-o", "ForwardAgent=yes"])

        if self.known_hosts_policy == "ignore":
            args.extend(["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null"])
        elif self.known_hosts_policy == "auto_add":
            args.extend(["-o", "StrictHostKeyChecking=accept-new"])

        if self.private_key_path:
            args.extend(["-i", self.private_key_path])

        for option, value in self.additional_options.items():
            args.extend(["-o", f"{option}={value}"])

        return args

    def get_connection_string(self) -> str:
        """Get SSH connection string.

        Returns
        -------
        str
            SSH connection string in format user@host:port

        Raises
        ------
        ValueError
            If host is not configured
        """
        if not self.host:
            raise ValueError("Host must be configured to get connection string")

        if self.port == 22:
            return f"{self.username}@{self.host}"
        else:
            return f"{self.username}@{self.host}:{self.port}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert SSH configuration to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of SSH configuration
        """
        return {
            "username": self.username,
            "private_key_path": self.private_key_path,
            "host": self.host,
            "port": self.port,
            "connection_timeout": self.connection_timeout,
            "command_timeout": self.command_timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "use_ssh_agent": self.use_ssh_agent,
            "forward_agent": self.forward_agent,
            "compression": self.compression,
            "keep_alive_interval": self.keep_alive_interval,
            "known_hosts_policy": self.known_hosts_policy,
            "additional_options": self.additional_options,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSHConfig":
        """Create SSH configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing SSH configuration parameters

        Returns
        -------
        SSHConfig
            SSH configuration instance
        """
        return cls(
            username=data.get("username", "ubuntu"),
            private_key_path=data.get("private_key_path"),
            host=data.get("host"),
            port=data.get("port", 22),
            connection_timeout=data.get("connection_timeout", 30),
            command_timeout=data.get("command_timeout", 300),
            max_retries=data.get("max_retries", 3),
            retry_delay=data.get("retry_delay", 5),
            use_ssh_agent=data.get("use_ssh_agent", True),
            forward_agent=data.get("forward_agent", False),
            compression=data.get("compression", True),
            keep_alive_interval=data.get("keep_alive_interval", 60),
            known_hosts_policy=data.get("known_hosts_policy", "auto_add"),
            additional_options=data.get("additional_options", {}),
        )

    @classmethod
    def for_development(cls, **kwargs) -> "SSHConfig":
        """Create SSH configuration optimized for development.

        Parameters
        ----------
        **kwargs
            Additional configuration parameters

        Returns
        -------
        SSHConfig
            Development-optimized SSH configuration
        """
        return cls(
            connection_timeout=10,
            command_timeout=120,
            max_retries=2,
            retry_delay=2,
            known_hosts_policy="auto_add",
            compression=True,
            **kwargs,
        )

    @classmethod
    def for_production(cls, **kwargs) -> "SSHConfig":
        """Create SSH configuration optimized for production.

        Parameters
        ----------
        **kwargs
            Additional configuration parameters

        Returns
        -------
        SSHConfig
            Production-optimized SSH configuration
        """
        return cls(
            connection_timeout=30,
            command_timeout=600,
            max_retries=5,
            retry_delay=10,
            known_hosts_policy="strict",
            compression=False,
            keep_alive_interval=30,
            **kwargs,
        )

    def update_host(self, host: str, port: Optional[int] = None) -> None:
        """Update the SSH host and optionally port.

        Parameters
        ----------
        host : str
            New SSH host/IP address
        port : Optional[int], optional
            New SSH port, by default None (keep current)
        """
        self.host = host
        if port is not None:
            self.port = port

    def create_private_key_file(self, key_content: str, file_path: str) -> None:
        """Create a private key file from content.

        Parameters
        ----------
        key_content : str
            Private key content
        file_path : str
            Path where to save the private key file
        """
        key_path = Path(file_path)
        key_path.parent.mkdir(parents=True, exist_ok=True)

        key_path.write_text(key_content)
        key_path.chmod(0o600)

        self.private_key_path = str(key_path)
        self.private_key_content = key_content
