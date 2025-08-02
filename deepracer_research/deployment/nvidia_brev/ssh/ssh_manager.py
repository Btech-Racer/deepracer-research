import subprocess
import time
from typing import Any, Dict, Optional, Tuple

import paramiko

from deepracer_research.deployment.nvidia_brev.config.ssh_config import SSHConfig
from deepracer_research.deployment.nvidia_brev.models import NvidiaBrevError
from deepracer_research.deployment.nvidia_brev.models.instance_models import InstanceDetails


class NvidiaBrevSSHManager:
    """SSH manager for NVIDIA Brev instances

    Parameters
    ----------
    ssh_config : SSHConfig
        SSH configuration settings
    instance : InstanceDetails, optional
        Instance details for connection, by default None
    """

    def __init__(self, ssh_config: SSHConfig, instance: Optional[InstanceDetails] = None):
        self.ssh_config = ssh_config
        self.instance = instance
        self._client = None

        if instance:
            self.ssh_config.update_host(instance.connection_host, instance.ssh_port)

    def connect(self, instance: Optional[InstanceDetails] = None) -> None:
        """Connect to the instance via SSH.

        Parameters
        ----------
        instance : Optional[InstanceDetails], optional
            Instance to connect to, by default None (use existing instance)

        Raises
        ------
        NvidiaBrevError
            If connection fails
        """
        if instance:
            self.instance = instance
            self.ssh_config.update_host(instance.connection_host, instance.ssh_port)

        if not self.instance or not self.instance.can_connect:
            raise NvidiaBrevError("Instance is not available for SSH connection")

        try:
            self._client = paramiko.SSHClient()

            if self.ssh_config.known_hosts_policy == "ignore":
                self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            elif self.ssh_config.known_hosts_policy == "auto_add":
                self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            else:
                self._client.load_system_host_keys()
                self._client.set_missing_host_key_policy(paramiko.RejectPolicy())

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

            for attempt in range(self.ssh_config.max_retries + 1):
                try:
                    self._client.connect(**connect_kwargs)
                    break
                except Exception as e:
                    if attempt < self.ssh_config.max_retries:
                        time.sleep(self.ssh_config.retry_delay)
                    else:
                        raise NvidiaBrevError(
                            f"SSH connection failed after {self.ssh_config.max_retries + 1} attempts: {str(e)}"
                        )

        except Exception as e:
            if self._client:
                self._client.close()
                self._client = None
            raise NvidiaBrevError(f"Failed to establish SSH connection: {str(e)}")

    def disconnect(self) -> None:
        """Disconnect SSH connection."""
        if self._client:
            self._client.close()
            self._client = None

    def is_connected(self) -> bool:
        """Check if SSH connection is active.

        Returns
        -------
        bool
            True if connected
        """
        if not self._client:
            return False

        try:
            transport = self._client.get_transport()
            return transport and transport.is_active()
        except:
            return False

    def execute_command(
        self, command: str, timeout: Optional[int] = None, get_pty: bool = False, environment: Optional[Dict[str, str]] = None
    ) -> Tuple[int, str, str]:
        """Execute a command on the remote instance.

        Parameters
        ----------
        command : str
            Command to execute
        timeout : Optional[int], optional
            Command timeout in seconds, by default None (use config timeout)
        get_pty : bool, optional
            Whether to allocate a pseudo-terminal, by default False
        environment : Optional[Dict[str, str]], optional
            Environment variables for the command, by default None

        Returns
        -------
        Tuple[int, str, str]
            Exit code, stdout, stderr

        Raises
        ------
        NvidiaBrevError
            If command execution fails
        """
        if not self.is_connected():
            self.connect()

        timeout = timeout or self.ssh_config.command_timeout

        try:
            if environment:
                env_str = " ".join([f"{k}='{v}'" for k, v in environment.items()])
                command = f"env {env_str} {command}"

            stdin, stdout, stderr = self._client.exec_command(
                command, timeout=timeout, get_pty=get_pty, environment=environment
            )

            exit_code = stdout.channel.recv_exit_status()
            stdout_text = stdout.read().decode("utf-8")
            stderr_text = stderr.read().decode("utf-8")

            return exit_code, stdout_text, stderr_text

        except Exception as e:
            raise NvidiaBrevError(f"Command execution failed: {str(e)}")

    def execute_script(
        self,
        script_content: str,
        script_name: str = "remote_script.sh",
        timeout: Optional[int] = None,
        make_executable: bool = True,
    ) -> Tuple[int, str, str]:
        """Execute a script on the remote instance.

        Parameters
        ----------
        script_content : str
            Script content to execute
        script_name : str, optional
            Name for the temporary script file, by default "remote_script.sh"
        timeout : Optional[int], optional
            Execution timeout, by default None
        make_executable : bool, optional
            Whether to make script executable, by default True

        Returns
        -------
        Tuple[int, str, str]
            Exit code, stdout, stderr
        """
        remote_path = f"/tmp/{script_name}"
        self.upload_content(script_content, remote_path)

        try:
            if make_executable:
                self.execute_command(f"chmod +x {remote_path}")

            return self.execute_command(f"bash {remote_path}", timeout=timeout)

        finally:
            try:
                self.execute_command(f"rm -f {remote_path}")
            except:
                pass

    def upload_file(self, local_path: str, remote_path: str) -> None:
        """Upload a file to the remote instance.

        Parameters
        ----------
        local_path : str
            Local file path
        remote_path : str
            Remote file path

        Raises
        ------
        NvidiaBrevError
            If upload fails
        """
        if not self.is_connected():
            self.connect()

        try:
            sftp = self._client.open_sftp()
            sftp.put(local_path, remote_path)
            sftp.close()
        except Exception as e:
            raise NvidiaBrevError(f"File upload failed: {str(e)}")

    def upload_content(self, content: str, remote_path: str, mode: int = 0o644) -> None:
        """Upload content as a file to the remote instance.

        Parameters
        ----------
        content : str
            Content to upload
        remote_path : str
            Remote file path
        mode : int, optional
            File permissions, by default 0o644

        Raises
        ------
        NvidiaBrevError
            If upload fails
        """
        if not self.is_connected():
            self.connect()

        try:
            sftp = self._client.open_sftp()
            with sftp.open(remote_path, "w") as remote_file:
                remote_file.write(content)
            sftp.chmod(remote_path, mode)
            sftp.close()
        except Exception as e:
            raise NvidiaBrevError(f"Content upload failed: {str(e)}")

    def download_file(self, remote_path: str, local_path: str) -> None:
        """Download a file from the remote instance.

        Parameters
        ----------
        remote_path : str
            Remote file path
        local_path : str
            Local file path

        Raises
        ------
        NvidiaBrevError
            If download fails
        """
        if not self.is_connected():
            self.connect()

        try:
            sftp = self._client.open_sftp()
            sftp.get(remote_path, local_path)
            sftp.close()
        except Exception as e:
            raise NvidiaBrevError(f"File download failed: {str(e)}")

    def file_exists(self, remote_path: str) -> bool:
        """Check if a file exists on the remote instance.

        Parameters
        ----------
        remote_path : str
            Remote file path

        Returns
        -------
        bool
            True if file exists
        """
        try:
            exit_code, _, _ = self.execute_command(f"test -f {remote_path}")
            return exit_code == 0
        except:
            return False

    def directory_exists(self, remote_path: str) -> bool:
        """Check if a directory exists on the remote instance.

        Parameters
        ----------
        remote_path : str
            Remote directory path

        Returns
        -------
        bool
            True if directory exists
        """
        try:
            exit_code, _, _ = self.execute_command(f"test -d {remote_path}")
            return exit_code == 0
        except:
            return False

    def create_directory(self, remote_path: str, mode: int = 0o755) -> None:
        """Create a directory on the remote instance.

        Parameters
        ----------
        remote_path : str
            Remote directory path
        mode : int, optional
            Directory permissions, by default 0o755
        """
        self.execute_command(f"mkdir -p {remote_path}")
        self.execute_command(f"chmod {oct(mode)[2:]} {remote_path}")

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information from the remote instance.

        Returns
        -------
        Dict[str, Any]
            System information
        """
        info = {}

        try:
            _, os_info, _ = self.execute_command("cat /etc/os-release")
            info["os_release"] = os_info.strip()

            _, cpu_info, _ = self.execute_command("nproc")
            info["cpu_cores"] = int(cpu_info.strip())

            _, mem_info, _ = self.execute_command("free -h")
            info["memory"] = mem_info.strip()

            _, gpu_info, _ = self.execute_command("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits")
            if gpu_info.strip():
                info["gpu"] = gpu_info.strip()

            _, docker_info, _ = self.execute_command("docker --version")
            if docker_info.strip():
                info["docker"] = docker_info.strip()

            _, disk_info, _ = self.execute_command("df -h /")
            info["disk"] = disk_info.strip()

        except Exception as e:
            info["error"] = str(e)

        return info

    def wait_for_service(self, service_name: str, port: int, timeout: int = 300, check_interval: int = 5) -> bool:
        """Wait for a service to become available.

        Parameters
        ----------
        service_name : str
            Name of the service
        port : int
            Port to check
        timeout : int, optional
            Maximum wait time in seconds, by default 300
        check_interval : int, optional
            Check interval in seconds, by default 5

        Returns
        -------
        bool
            True if service becomes available
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                exit_code, _, _ = self.execute_command(f"netstat -tuln | grep :{port}")
                if exit_code == 0:
                    return True
            except:
                pass

            time.sleep(check_interval)

        return False

    def port_forward(self, local_port: int, remote_port: int, local_host: str = "localhost") -> subprocess.Popen:
        """Create SSH port forward.

        Parameters
        ----------
        local_port : int
            Local port
        remote_port : int
            Remote port
        local_host : str, optional
            Local host, by default "localhost"

        Returns
        -------
        subprocess.Popen
            SSH process for port forwarding

        Raises
        ------
        NvidiaBrevError
            If port forwarding fails
        """
        if not self.ssh_config.host:
            raise NvidiaBrevError("SSH host not configured")

        ssh_cmd = [
            "ssh",
            "-N",
            "-L",
            f"{local_host}:{local_port}:{local_host}:{remote_port}",
            f"{self.ssh_config.username}@{self.ssh_config.host}",
            "-p",
            str(self.ssh_config.port),
        ]

        ssh_cmd.extend(self.ssh_config.ssh_command_args)

        try:
            process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            time.sleep(2)

            if process.poll() is not None:
                stdout, stderr = process.communicate()
                raise NvidiaBrevError(f"Port forwarding failed: {stderr.decode()}")

            return process

        except Exception as e:
            raise NvidiaBrevError(f"Failed to create port forward: {str(e)}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
