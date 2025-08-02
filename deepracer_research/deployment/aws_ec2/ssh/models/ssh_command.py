from dataclasses import dataclass
from typing import Optional


@dataclass
class SSHCommand:
    """Represents an SSH command executed on an EC2 instance"""

    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: Optional[int] = None
    execution_time: float = 0.0
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Check if command executed successfully.

        Returns
        -------
        bool
            True if command completed with exit code 0.
        """
        return self.exit_code == 0 and not self.timed_out

    @property
    def failed(self) -> bool:
        """Check if command failed.

        Returns
        -------
        bool
            True if command failed or timed out.
        """
        return not self.success

    @property
    def output(self) -> str:
        """Get combined stdout and stderr output.

        Returns
        -------
        str
            Combined output from the command.
        """
        output_parts = []
        if self.stdout:
            output_parts.append(self.stdout)
        if self.stderr:
            output_parts.append(self.stderr)
        return "\n".join(output_parts)

    def __str__(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        return f"SSHCommand('{self.command}') -> {status} (exit_code={self.exit_code})"
