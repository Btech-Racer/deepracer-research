from dataclasses import dataclass
from typing import Any, Dict, Optional

from deepracer_research.config.aws.aws_hyperparameters import AWSHyperparameters
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm


@dataclass
class DeepRacerHyperparameters(AWSHyperparameters):
    """Hyperparameters for AWS DeepRacer training

    Parameters
    ----------
    beta : float, optional
        PPO-specific parameter for policy optimization, by default 0.01
    sac_alpha : float, optional
        SAC-specific parameter for entropy regularization, by default 0.2
    """

    beta: float = 0.01
    sac_alpha: float = 0.2

    def __post_init__(self):
        """Post-initialization to apply DeepRacer-specific defaults."""
        super().__post_init__() if hasattr(super(), "__post_init__") else None

        self._apply_deepracer_defaults()

    def _apply_deepracer_defaults(self):
        """Apply DeepRacer console optimized defaults."""
        if self.num_epochs == 10:
            self.num_epochs = 3

        if self.stack_size == 1:
            self.stack_size = 3

    @classmethod
    def from_aws_hyperparameters(cls, aws_params: AWSHyperparameters, **deepracer_overrides) -> "DeepRacerHyperparameters":
        """Create DeepRacer hyperparameters from AWS hyperparameters.

        Parameters
        ----------
        aws_params : AWSHyperparameters
            Base AWS hyperparameters to map from
        **deepracer_overrides
            DeepRacer-specific parameter overrides

        Returns
        -------
        DeepRacerHyperparameters
            Mapped DeepRacer hyperparameters
        """
        aws_dict = aws_params.to_dict()

        deepracer_defaults = {"num_epochs": 3, "stack_size": 3, "beta": 0.01, "sac_alpha": 0.2}

        final_params = {**aws_dict, **deepracer_defaults, **deepracer_overrides}

        return cls(**final_params)

    @classmethod
    def create_for_algorithm(
        cls, algorithm: TrainingAlgorithm, base_params: Optional[AWSHyperparameters] = None, **overrides
    ) -> "DeepRacerHyperparameters":
        """Create algorithm-optimized hyperparameters.

        Parameters
        ----------
        algorithm : TrainingAlgorithm
            Training algorithm (TrainingAlgorithm.PPO or TrainingAlgorithm.SAC)
        base_params : Optional[AWSHyperparameters], optional
            Base parameters to start from, by default None
        **overrides
            Additional parameter overrides

        Returns
        -------
        DeepRacerHyperparameters
            Algorithm-optimized hyperparameters
        """
        if base_params is None:
            from deepracer_research.config.aws.aws_hyperparameters import DEFAULT_HYPERPARAMETERS

            base_params = DEFAULT_HYPERPARAMETERS

        algorithm_defaults = {
            TrainingAlgorithm.PPO: {
                "learning_rate": 0.0003,
                "entropy_coefficient": 0.01,
                "epsilon": 0.2,
                "beta": 0.01,
                "num_epochs": 3,
            },
            TrainingAlgorithm.SAC: {"learning_rate": 0.0003, "entropy_coefficient": 0.02, "sac_alpha": 0.2, "num_epochs": 5},
        }

        algo_params = algorithm_defaults.get(algorithm.lower(), {})
        final_overrides = {**algo_params, **overrides}

        return cls.from_aws_hyperparameters(base_params, **final_overrides)

    def to_deepracer_format(self) -> Dict[str, str]:
        """Convert to AWS DeepRacer console format (all values as strings).

        Returns
        -------
        Dict[str, str]
            Hyperparameters formatted for DeepRacer API
        """
        base_dict = super().to_dict()

        deepracer_dict = {**base_dict, "beta": str(self.beta), "sac_alpha": str(self.sac_alpha)}

        return {key: str(value) for key, value in deepracer_dict.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format with proper types.

        Returns
        -------
        Dict[str, Any]
            Hyperparameters as dictionary with proper types
        """
        base_dict = super().to_dict()

        return {**base_dict, "beta": self.beta, "sac_alpha": self.sac_alpha}

    def validate(self) -> bool:
        """Validate hyperparameters including DeepRacer-specific ones.

        Returns
        -------
        bool
            True if hyperparameters are valid
        """
        if not super().validate():
            return False

        if not 0 <= self.beta <= 1:
            return False
        if not 0 <= self.sac_alpha <= 1:
            return False

        if self.stack_size < 1 or self.stack_size > 5:
            return False
        if self.num_epochs < 1 or self.num_epochs > 20:
            return False

        return True

    def get_algorithm_specific_params(self, algorithm: TrainingAlgorithm) -> Dict[str, Any]:
        """Get parameters specific to the given algorithm.

        Parameters
        ----------
        algorithm : TrainingAlgorithm
            Algorithm name ("ppo" or "sac")

        Returns
        -------
        Dict[str, Any]
            Algorithm-specific parameters
        """
        if algorithm == TrainingAlgorithm.PPO:
            return {"epsilon": self.epsilon, "beta": self.beta, "beta_entropy": self.beta_entropy}
        elif algorithm == TrainingAlgorithm.SAC:
            return {"sac_alpha": self.sac_alpha, "entropy_coefficient": self.entropy_coefficient}
        else:
            return {}


def create_default_deepracer_hyperparameters() -> DeepRacerHyperparameters:
    """Create default DeepRacer hyperparameters.

    Returns
    -------
    DeepRacerHyperparameters
        Default configuration optimized for DeepRacer console
    """
    from deepracer_research.config.aws.aws_hyperparameters import DEFAULT_HYPERPARAMETERS

    return DeepRacerHyperparameters.from_aws_hyperparameters(DEFAULT_HYPERPARAMETERS)


def create_ppo_optimized_hyperparameters(base_params: Optional[AWSHyperparameters] = None) -> DeepRacerHyperparameters:
    """Create PPO-optimized hyperparameters.

    Parameters
    ----------
    base_params : Optional[AWSHyperparameters], optional
        Base parameters to start from, by default None

    Returns
    -------
    DeepRacerHyperparameters
        PPO-optimized configuration
    """
    return DeepRacerHyperparameters.create_for_algorithm("ppo", base_params)


def create_sac_optimized_hyperparameters(base_params: Optional[AWSHyperparameters] = None) -> DeepRacerHyperparameters:
    """Create SAC-optimized hyperparameters.

    Parameters
    ----------
    base_params : Optional[AWSHyperparameters], optional
        Base parameters to start from, by default None

    Returns
    -------
    DeepRacerHyperparameters
        SAC-optimized configuration
    """
    return DeepRacerHyperparameters.create_for_algorithm("sac", base_params)


def map_scenario_to_hyperparameters(
    scenario: str, algorithm: TrainingAlgorithm = TrainingAlgorithm.PPO, base_params: Optional[AWSHyperparameters] = None
) -> DeepRacerHyperparameters:
    """Map experimental scenario to optimized hyperparameters.

    Parameters
    ----------
    scenario : str
        Experimental scenario name
    algorithm : TrainingAlgorithm, optional
        Training algorithm, by default TrainingAlgorithm.PPO
    base_params : Optional[AWSHyperparameters], optional
        Base parameters to start from, by default None

    Returns
    -------
    DeepRacerHyperparameters
        Scenario-optimized hyperparameters
    """
    scenario_optimizations = {
        "speed_optimization": {"learning_rate": 0.0005, "entropy_coefficient": 0.02, "discount_factor": 0.95},
        "centerline_following": {"learning_rate": 0.0003, "entropy_coefficient": 0.01, "discount_factor": 0.999},
        "time_trial": {"learning_rate": 0.001, "entropy_coefficient": 0.015, "discount_factor": 0.99},
    }

    scenario_params = scenario_optimizations.get(scenario.lower(), {})
    return DeepRacerHyperparameters.create_for_algorithm(algorithm, base_params, **scenario_params)


DEFAULT_DEEPRACER_HYPERPARAMETERS = create_default_deepracer_hyperparameters()
