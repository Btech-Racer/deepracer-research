from dataclasses import dataclass


@dataclass
class SSHCommand:
    """SSH command execution result"""

    command: str
    stdout: str
    stderr: str
    return_code: int
    success: bool
