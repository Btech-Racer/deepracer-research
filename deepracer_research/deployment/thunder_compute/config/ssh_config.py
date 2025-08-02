from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class SSHConfig:
    """SSH configuration for Thunder Compute instances

    Parameters
    ----------
    use_tnr_cli : bool, optional
        Whether to use Thunder Compute CLI for SSH connections, by default True.
    ssh_key_path : str, optional
        Path to SSH private key file, by default None.
    port : int, optional
        SSH port number, by default 22.
    username : str, optional
        SSH username, by default "ubuntu".
    connect_timeout : int, optional
        SSH connection timeout in seconds, by default 30.
    """

    use_tnr_cli: bool = True
    ssh_key_path: Optional[str] = None
    port: int = 22
    username: str = "ubuntu"
    connect_timeout: int = 30

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the configuration.
        """
        return {
            "use_tnr_cli": self.use_tnr_cli,
            "ssh_key_path": self.ssh_key_path,
            "port": self.port,
            "username": self.username,
            "connect_timeout": self.connect_timeout,
        }

    def validate(self) -> None:
        """Validate the configuration parameters.

        Raises
        ------
        ValueError
            If any configuration parameter is invalid.
        """
        if self.port <= 0 or self.port > 65535:
            raise ValueError("SSH port must be between 1 and 65535")

        if self.connect_timeout <= 0:
            raise ValueError("Connect timeout must be positive")

        if not self.username:
            raise ValueError("Username is required")

    @classmethod
    def for_thunder_cli(cls) -> "SSHConfig":
        """Create configuration for Thunder Compute CLI connections.

        Returns
        -------
        SSHConfig
            SSH configuration optimized for Thunder CLI usage.
        """
        return cls(use_tnr_cli=True, port=22, username="ubuntu", connect_timeout=30)

    @classmethod
    def for_manual_ssh(cls, ssh_key_path: str, username: str = "ubuntu", port: int = 22) -> "SSHConfig":
        """Create configuration for manual SSH connections.

        Parameters
        ----------
        ssh_key_path : str
            Path to SSH private key file.
        username : str, optional
            SSH username, by default "ubuntu".
        port : int, optional
            SSH port number, by default 22.

        Returns
        -------
        SSHConfig
            SSH configuration for manual connections.
        """
        return cls(use_tnr_cli=False, ssh_key_path=ssh_key_path, port=port, username=username, connect_timeout=30)

    @classmethod
    def skip_tnr_cli(cls, username: str = "ubuntu", port: int = 22) -> "SSHConfig":
        """Create configuration that skips TNR CLI and uses direct SSH.

        Parameters
        ----------
        username : str, optional
            SSH username, by default "ubuntu".
        port : int, optional
            SSH port number, by default 22.

        Returns
        -------
        SSHConfig
            SSH configuration that bypasses TNR CLI setup.
        """
        return cls(use_tnr_cli=False, ssh_key_path=None, port=port, username=username, connect_timeout=30)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SSHConfig":
        """Create configuration from dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing configuration parameters.

        Returns
        -------
        SSHConfig
            SSH configuration instance created from dictionary.
        """
        return cls(
            use_tnr_cli=data.get("use_tnr_cli", True),
            ssh_key_path=data.get("ssh_key_path"),
            port=data.get("port", 22),
            username=data.get("username", "ubuntu"),
            connect_timeout=data.get("connect_timeout", 30),
        )
