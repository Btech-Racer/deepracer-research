from typing import Optional


class SSHConnectionError(Exception):
    """Exception raised when SSH connection to EC2 instance fails"""

    def __init__(
        self, message: str, instance_id: Optional[str] = None, hostname: Optional[str] = None, port: Optional[int] = None
    ):
        super().__init__(message)
        self.message = message
        self.instance_id = instance_id
        self.hostname = hostname
        self.port = port

    def __str__(self) -> str:
        base_message = self.message
        if self.instance_id:
            base_message += f" (Instance: {self.instance_id})"
        if self.hostname:
            base_message += f" (Host: {self.hostname})"
        if self.port:
            base_message += f" (Port: {self.port})"
        return base_message
