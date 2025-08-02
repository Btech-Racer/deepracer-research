from enum import Enum, unique

from deepracer_research.config.training.training_algorithm import TrainingAlgorithm


@unique
class ActionSpaceType(str, Enum):
    """Extended action space types for local DeepRacer training"""

    CONTINUOUS = "continuous"
    DISCRETE = "discrete"
    HYBRID = "hybrid"
    MULTI_DISCRETE = "multi_discrete"

    @classmethod
    def get_recommended_for_algorithm(cls, algorithm: str) -> "ActionSpaceType":
        """Get recommended action space type for a training algorithm.

        Parameters
        ----------
        algorithm : str
            Training algorithm name

        Returns
        -------
        ActionSpaceType
            Recommended action space type
        """
        discrete_algorithms = {TrainingAlgorithm.PPO, TrainingAlgorithm.DQN, TrainingAlgorithm.RAINBOW_DQN}
        continuous_algorithms = {TrainingAlgorithm.SAC, TrainingAlgorithm.TD3, TrainingAlgorithm.DDPG}

        algorithm_lower = algorithm.lower()

        if algorithm_lower in discrete_algorithms:
            return cls.DISCRETE
        elif algorithm_lower in continuous_algorithms:
            return cls.CONTINUOUS
        else:
            return cls.DISCRETE

    def get_description(self) -> str:
        """Get a  description of the action space type.

        Returns
        -------
        str
            Description of the action space type
        """
        descriptions = {
            self.DISCRETE: "Finite set of predefined actions",
            self.CONTINUOUS: "Real-valued actions",
            self.HYBRID: "Mix of discrete and continuous actions",
            self.MULTI_DISCRETE: "Multiple discrete action dimensions",
        }
        return descriptions[self]

    def supports_algorithm(self, algorithm: str) -> bool:
        """Check if this action space type is compatible with an algorithm.

        Parameters
        ----------
        algorithm : str
            Training algorithm name

        Returns
        -------
        bool
            True if compatible
        """
        algorithm_lower = algorithm.lower()

        if self == self.DISCRETE:
            return algorithm_lower in {TrainingAlgorithm.PPO, TrainingAlgorithm.DQN, TrainingAlgorithm.RAINBOW_DQN}
        elif self == self.CONTINUOUS:
            return algorithm_lower in {
                TrainingAlgorithm.SAC,
                TrainingAlgorithm.TD3,
                TrainingAlgorithm.DDPG,
                TrainingAlgorithm.PPO,
            }
        elif self == self.HYBRID:
            return algorithm_lower in {TrainingAlgorithm.PPO, TrainingAlgorithm.SAC}
        elif self == self.MULTI_DISCRETE:
            return algorithm_lower in {TrainingAlgorithm.PPO, TrainingAlgorithm.A2C, TrainingAlgorithm.DQN}
        else:
            return False
