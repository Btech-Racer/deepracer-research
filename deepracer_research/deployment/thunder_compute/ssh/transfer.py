import os
import subprocess
from typing import Optional

from deepracer_research.deployment.thunder_compute.config.ssh_config import SSHConfig
from deepracer_research.deployment.thunder_compute.ssh.models.ssh_connection_error import SSHConnectionError
from deepracer_research.utils import error, info, warning


class SSHFileTransfer:
    """File transfer manager for SSH connections to Thunder Compute instances"""

    def __init__(self, instance_uuid: str, ssh_config: Optional[SSHConfig] = None, thunder_cli_index: Optional[str] = None):
        """Initialize SSH file transfer manager.

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

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload file to remote instance.

        Parameters
        ------------
        local_path : str
            Local file path.
        remote_path : str
            Remote file path.

        Returns
        -------
        bool
            True if upload successful.

        Raises
        ------
        SSHConnectionError
            If file upload fails.
        """
        if not os.path.exists(local_path):
            raise SSHConnectionError(f"Local file not found: {local_path}")

        if hasattr(self, "thunder_cli_index") and self.thunder_cli_index is not None:
            info(
                "Using TNR scp for file upload",
                extra={"instance_id": self.thunder_cli_index, "local_path": local_path, "remote_path": remote_path},
            )

            tnr_command = ["tnr", "scp", local_path, f"{self.thunder_cli_index}:{remote_path}"]

            try:
                result = subprocess.run(tnr_command, capture_output=True, text=True, timeout=300)

                if result.returncode == 0:
                    info("File upload successful")
                    return True
                else:
                    error("TNR scp file upload failed", extra={"stderr": result.stderr})
                    raise SSHConnectionError(f"TNR scp file upload failed: {result.stderr}")

            except subprocess.TimeoutExpired:
                raise SSHConnectionError("TNR scp file upload timed out")
            except Exception as e:
                raise SSHConnectionError(f"TNR scp file upload error: {e}")

        elif self.ssh_config.use_tnr_cli:
            scp_command = ["scp", local_path, f"{self.tnr_alias}:{remote_path}"]
        else:
            scp_command = ["scp", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no"]

            if self.ssh_config.ssh_key_path:
                scp_command.extend(["-i", self.ssh_config.ssh_key_path])

            scp_command.extend([local_path, f"{self.ssh_config.username}@{self.instance_uuid}:{remote_path}"])

        try:
            info("Using scp fallback for file upload", extra={"local_path": local_path, "remote_path": remote_path})
            result = subprocess.run(scp_command, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                info("File upload successful")
                return True
            else:
                error("File upload failed", extra={"stderr": result.stderr})
                raise SSHConnectionError(f"File upload failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise SSHConnectionError("File upload timed out")
        except Exception as e:
            raise SSHConnectionError(f"File upload error: {e}")

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download file from remote instance.

        Parameters
        ------------
        remote_path : str
            Remote file path.
        local_path : str
            Local file path.

        Returns
        -------
        bool
            True if download successful.

        Raises
        ------
        SSHConnectionError
            If file download fails.
        """
        local_dir = os.path.dirname(local_path)
        if local_dir:
            os.makedirs(local_dir, exist_ok=True)

        if self.ssh_config.use_tnr_cli:
            scp_command = ["scp", f"{self.tnr_alias}:{remote_path}", local_path]
        else:
            scp_command = ["scp", "-o", "ConnectTimeout=30", "-o", "StrictHostKeyChecking=no"]

            if self.ssh_config.ssh_key_path:
                scp_command.extend(["-i", self.ssh_config.ssh_key_path])

            scp_command.extend([f"{self.ssh_config.username}@{self.instance_uuid}:{remote_path}", local_path])

        try:
            info("Downloading file", extra={"remote_path": remote_path, "local_path": local_path})
            result = subprocess.run(scp_command, capture_output=True, text=True, timeout=300)

            if result.returncode == 0:
                info("File download successful")
                return True
            else:
                error("File download failed", extra={"stderr": result.stderr})
                raise SSHConnectionError(f"File download failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise SSHConnectionError("File download timed out")
        except Exception as e:
            raise SSHConnectionError(f"File download error: {e}")

    def upload_directory(self, local_dir: str, remote_dir: str, exclude_patterns: Optional[list] = None) -> bool:
        """Upload directory to remote instance with rsync.

        Parameters
        ----------
        local_dir : str
            Local directory path.
        remote_dir : str
            Remote directory path.
        exclude_patterns : list, optional
            Patterns to exclude (e.g., ['.git/', '__pycache__/', '*.pyc']).

        Returns
        -------
        bool
            True if upload successful.

        Raises
        ------
        SSHConnectionError
            If directory upload fails.
        """
        if not os.path.exists(local_dir):
            raise SSHConnectionError(f"Local directory not found: {local_dir}")

        default_exclusions = [
            ".git/",
            "__pycache__/",
            "*.pyc",
            "*.pyo",
            ".pytest_cache/",
            ".mypy_cache/",
            "node_modules/",
            ".DS_Store",
            ".env",
            "venv/",
            ".venv/",
            "env/",
            ".coverage",
            "*.log",
            "logs/",
            "tmp/",
            "temp/",
            "*.egg-info/",
            "dist/",
            "build/",
        ]

        exclusions = default_exclusions + (exclude_patterns or [])

        if hasattr(self, "thunder_cli_index") and self.thunder_cli_index is not None:
            local_dir_abs = os.path.abspath(local_dir)
            if not os.path.exists(local_dir_abs):
                error("Local directory not found", extra={"path": local_dir_abs})
                raise SSHConnectionError(f"Local directory not found: {local_dir_abs}")

            info(
                "Using TNR scp for directory upload",
                extra={"instance_id": self.thunder_cli_index, "local_dir": local_dir_abs, "remote_dir": remote_dir},
            )

            tnr_command = ["tnr", "scp", local_dir_abs, f"{self.thunder_cli_index}:{remote_dir}"]

            try:
                info("Executing TNR scp command", extra={"command": " ".join(tnr_command)})
                result = subprocess.run(tnr_command, capture_output=True, text=True, timeout=600, stdin=subprocess.DEVNULL)

                if result.returncode == 0:
                    info("Directory upload successful")
                    return True
                else:
                    error("TNR scp upload failed", extra={"stderr": result.stderr, "stdout": result.stdout})
                    warning("TNR scp failed, falling back to rsync")

            except subprocess.TimeoutExpired:
                warning("TNR scp upload timed out, falling back to rsync")
            except Exception as e:
                warning(f"TNR scp upload error: {e}, falling back to rsync")

        if hasattr(self, "thunder_cli_index") and self.thunder_cli_index is not None:
            rsync_command = ["rsync", "-avz", "--progress", "--rsh", "ssh"]

            for pattern in exclusions:
                rsync_command.extend(["--exclude", pattern])

            local_dir_rsync = os.path.abspath(local_dir)
            if not local_dir_rsync.endswith("/"):
                local_dir_rsync += "/"

            destination = f"tnr-{self.thunder_cli_index}:{remote_dir}"
            rsync_command.extend([local_dir_rsync, destination])

        elif self.ssh_config.use_tnr_cli:
            rsync_command = ["rsync", "-avz", "--progress", "--rsh", "ssh"]

            for pattern in exclusions:
                rsync_command.extend(["--exclude", pattern])

            local_dir_rsync = os.path.abspath(local_dir)
            if not local_dir_rsync.endswith("/"):
                local_dir_rsync += "/"
            rsync_command.extend([local_dir_rsync, f"{self.tnr_alias}:{remote_dir}"])

        else:
            rsync_command = ["rsync", "-avz", "--progress", "-e", "ssh -o ConnectTimeout=30 -o StrictHostKeyChecking=no"]

            if self.ssh_config.ssh_key_path:
                rsync_command[rsync_command.index("-e")] = (
                    f"ssh -i {self.ssh_config.ssh_key_path} -o ConnectTimeout=30 -o StrictHostKeyChecking=no"
                )

            for pattern in exclusions:
                rsync_command.extend(["--exclude", pattern])

            local_dir_rsync = os.path.abspath(local_dir)
            if not local_dir_rsync.endswith("/"):
                local_dir_rsync += "/"

            destination = f"{self.ssh_config.username}@{self.instance_uuid}:{remote_dir}"
            rsync_command.extend([local_dir_rsync, destination])

        try:
            info(
                "Using rsync for directory upload",
                extra={
                    "local_dir": os.path.abspath(local_dir),
                    "remote_dir": remote_dir,
                    "exclusions": len(exclusions),
                    "command": " ".join(rsync_command[:5]),
                },
            )

            result = subprocess.run(rsync_command, capture_output=True, text=True, timeout=1800)

            if result.returncode == 0:
                info("Directory upload successful")
                return True
            else:
                error("Directory upload failed", extra={"stderr": result.stderr})
                raise SSHConnectionError(f"Directory upload failed: {result.stderr}")

        except subprocess.TimeoutExpired:
            raise SSHConnectionError("Directory upload timed out")
        except Exception as e:
            raise SSHConnectionError(f"Directory upload error: {e}")

    def sync_project(self, project_root: str, remote_project_dir: str = "~/deepracer-research") -> bool:
        """Sync entire DeepRacer research project to remote instance.

        Parameters
        ----------
        project_root : str
            Local project root directory.
        remote_project_dir : str, optional
            Remote project directory, by default "~/deepracer-research".

        Returns
        -------
        bool
            True if sync successful.
        """
        deepracer_exclusions = [
            "exported_models/",
            "imported_models/",
            "evaluation_videos/",
            "training_jobs/",
            "experiments/",
            "logs/",
            "local_deepracer/",
            ".ipynb_checkpoints/",
            "instance_*.json",
            "deployments/",
        ]

        info("Syncing DeepRacer research project", extra={"project_root": project_root, "remote_dir": remote_project_dir})

        return self.upload_directory(project_root, remote_project_dir, deepracer_exclusions)
