from enum import Enum


class LocalTrainingBackend(str, Enum):
    """Supported local training backends for DeepRacer model training"""

    TENSORFLOW = "tensorflow"
    PYTORCH = "pytorch"
    STABLE_BASELINES3 = "stable_baselines3"
    RAY_RLLIB = "ray_rllib"
    DEEPRACER_FOR_CLOUD = "deepracer_for_cloud"

    @classmethod
    def get_recommended_for_scenario(cls, scenario: str) -> "LocalTrainingBackend":
        """Get recommended backend for a specific training scenario.

        Parameters
        ----------
        scenario : str
            Training scenario name

        Returns
        -------
        LocalTrainingBackend
            Recommended backend for the scenario
        """
        recommendations = {
            "beginner": cls.STABLE_BASELINES3,
            "research": cls.PYTORCH,
            "production": cls.STABLE_BASELINES3,
            "scalable": cls.RAY_RLLIB,
            "custom": cls.TENSORFLOW,
            "aws_compatible": cls.DEEPRACER_FOR_CLOUD,
            "deepracer": cls.DEEPRACER_FOR_CLOUD,
        }
        return recommendations.get(scenario, cls.DEEPRACER_FOR_CLOUD)

    def get_description(self) -> str:
        """Get a  description of the backend.

        Returns
        -------
        str
            Description of the backend
        """
        descriptions = {
            self.TENSORFLOW: "Low-level TensorFlow for custom implementations",
            self.PYTORCH: "Dynamic PyTorch for research and prototyping",
            self.STABLE_BASELINES3: "High-level RL library for production use",
            self.RAY_RLLIB: "Distributed RL framework for scalable training",
            self.DEEPRACER_FOR_CLOUD: "AWS DeepRacer compatible local training environment",
        }
        return descriptions[self]

    def requires_gpu(self) -> bool:
        """Check if this backend benefits significantly from GPU acceleration.

        Returns
        -------
        bool
            True if GPU is recommended for this backend
        """
        gpu_backends = {self.TENSORFLOW, self.PYTORCH, self.RAY_RLLIB, self.DEEPRACER_FOR_CLOUD}
        return self in gpu_backends
