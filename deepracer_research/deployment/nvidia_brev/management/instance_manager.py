from deepracer_research.deployment.nvidia_brev.api.client import NvidiaBrevClient
from deepracer_research.deployment.nvidia_brev.ssh.ssh_manager import NvidiaBrevSSHManager


class InstanceManager:
    """Instance management for NVIDIA Brev instances"""

    def __init__(self, client: NvidiaBrevClient, ssh_manager: NvidiaBrevSSHManager):
        self.client = client
        self.ssh_manager = ssh_manager
