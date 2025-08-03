import os
from pathlib import Path
from typing import List, Optional

from deepracer_research.deployment.aws_ec2.config.ssh_config import EC2SSHConfig
from deepracer_research.deployment.aws_ec2.ssh.connection import SSHConnectionManager
from deepracer_research.utils.logger import debug, error, info, warning


class SSHFileTransfer:
    """Handles file transfers to/from EC2 instances via SSH"""

    def __init__(self, instance_id: str, hostname: str, ssh_config: Optional[EC2SSHConfig] = None):
        """Initialize SSH file transfer manager.

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
        self.connection_manager = SSHConnectionManager(instance_id, hostname, ssh_config)

    def upload_file(self, local_path: str, remote_path: str) -> bool:
        """Upload a file to the EC2 instance.

        Parameters
        ----------
        local_path : str
            Local file path.
        remote_path : str
            Remote file path.

        Returns
        -------
        bool
            True if upload was successful.
        """
        if not self.connection_manager.connect():
            error("Cannot upload file: SSH connection failed", extra={"instance_id": self.instance_id})
            return False

        try:
            info(
                "Uploading file", extra={"instance_id": self.instance_id, "local_path": local_path, "remote_path": remote_path}
            )

            sftp = self.connection_manager.client.open_sftp()

            remote_dir = os.path.dirname(remote_path)
            if remote_dir:
                self._create_remote_directory(sftp, remote_dir)

            sftp.put(local_path, remote_path)
            sftp.close()

            info(
                "File upload completed",
                extra={"instance_id": self.instance_id, "local_path": local_path, "remote_path": remote_path},
            )

            return True

        except Exception as e:
            error(
                "File upload failed",
                extra={"instance_id": self.instance_id, "local_path": local_path, "remote_path": remote_path, "error": str(e)},
            )
            return False

    def download_file(self, remote_path: str, local_path: str) -> bool:
        """Download a file from the EC2 instance.

        Parameters
        ----------
        remote_path : str
            Remote file path.
        local_path : str
            Local file path.

        Returns
        -------
        bool
            True if download was successful.
        """
        if not self.connection_manager.connect():
            error("Cannot download file: SSH connection failed", extra={"instance_id": self.instance_id})
            return False

        try:
            info(
                "Downloading file",
                extra={"instance_id": self.instance_id, "remote_path": remote_path, "local_path": local_path},
            )

            local_dir = os.path.dirname(local_path)
            if local_dir:
                os.makedirs(local_dir, exist_ok=True)

            sftp = self.connection_manager.client.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()

            info(
                "File download completed",
                extra={"instance_id": self.instance_id, "remote_path": remote_path, "local_path": local_path},
            )

            return True

        except Exception as e:
            error(
                "File download failed",
                extra={"instance_id": self.instance_id, "remote_path": remote_path, "local_path": local_path, "error": str(e)},
            )
            return False

    def upload_directory(self, local_dir: str, remote_dir: str, exclude_patterns: Optional[List[str]] = None) -> bool:
        """Upload a directory to the EC2 instance.

        Parameters
        ----------
        local_dir : str
            Local directory path.
        remote_dir : str
            Remote directory path.
        exclude_patterns : List[str], optional
            Patterns to exclude from upload, by default None.

        Returns
        -------
        bool
            True if upload was successful.
        """
        if not self.connection_manager.connect():
            error("Cannot upload directory: SSH connection failed", extra={"instance_id": self.instance_id})
            return False

        try:
            info(
                "Uploading directory", extra={"instance_id": self.instance_id, "local_dir": local_dir, "remote_dir": remote_dir}
            )

            exclude_patterns = exclude_patterns or []
            local_path = Path(local_dir)

            if not local_path.exists():
                error("Local directory does not exist", extra={"local_dir": local_dir})
                return False

            sftp = self.connection_manager.client.open_sftp()

            self._create_remote_directory(sftp, remote_dir)

            for root, dirs, files in os.walk(local_dir):
                rel_path = os.path.relpath(root, local_dir)
                if rel_path == ".":
                    remote_root = remote_dir
                else:
                    remote_root = os.path.join(remote_dir, rel_path).replace("\\", "/")

                dirs[:] = [d for d in dirs if not self._should_exclude(d, exclude_patterns)]

                if rel_path != ".":
                    self._create_remote_directory(sftp, remote_root)

                for file in files:
                    if self._should_exclude(file, exclude_patterns):
                        continue

                    local_file = os.path.join(root, file)
                    remote_file = os.path.join(remote_root, file).replace("\\", "/")

                    try:
                        sftp.put(local_file, remote_file)
                        debug("Uploaded file", extra={"local_file": local_file, "remote_file": remote_file})
                    except Exception as e:
                        warning(
                            "Failed to upload file",
                            extra={"local_file": local_file, "remote_file": remote_file, "error": str(e)},
                        )

            sftp.close()

            info(
                "Directory upload completed",
                extra={"instance_id": self.instance_id, "local_dir": local_dir, "remote_dir": remote_dir},
            )

            return True

        except Exception as e:
            error(
                "Directory upload failed",
                extra={"instance_id": self.instance_id, "local_dir": local_dir, "remote_dir": remote_dir, "error": str(e)},
            )
            return False

    def sync_project(self, project_root: str, remote_project_dir: str = "~/deepracer-research") -> bool:
        """Sync DeepRacer research project to EC2 instance.

        Parameters
        ----------
        project_root : str
            Local project root directory.
        remote_project_dir : str, optional
            Remote project directory, by default "~/deepracer-research".

        Returns
        -------
        bool
            True if sync was successful.
        """
        info(
            "Syncing DeepRacer project",
            extra={"instance_id": self.instance_id, "project_root": project_root, "remote_project_dir": remote_project_dir},
        )

        exclude_patterns = [
            "__pycache__",
            "*.pyc",
            ".git",
            ".pytest_cache",
            "*.log",
            ".DS_Store",
            "Thumbs.db",
            "node_modules",
            ".env",
            "*.tmp",
            "*.swp",
            ".vscode",
            ".idea",
        ]

        return self.upload_directory(project_root, remote_project_dir, exclude_patterns)

    def _create_remote_directory(self, sftp, remote_dir: str) -> None:
        """Create a directory on the remote instance if it doesn't exist.

        Parameters
        ----------
        sftp : paramiko.SFTPClient
            SFTP client.
        remote_dir : str
            Remote directory path.
        """
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError:
            parent_dir = os.path.dirname(remote_dir)
            if parent_dir and parent_dir != "/":
                self._create_remote_directory(sftp, parent_dir)

            try:
                sftp.mkdir(remote_dir)
                debug("Created remote directory", extra={"remote_dir": remote_dir})
            except Exception as e:
                debug("Failed to create remote directory", extra={"remote_dir": remote_dir, "error": str(e)})

    def _should_exclude(self, name: str, exclude_patterns: List[str]) -> bool:
        """Check if a file/directory should be excluded.

        Parameters
        ----------
        name : str
            File or directory name.
        exclude_patterns : List[str]
            List of patterns to exclude.

        Returns
        -------
        bool
            True if should be excluded.
        """
        import fnmatch

        for pattern in exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return True
        return False
