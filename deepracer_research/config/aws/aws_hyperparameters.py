from dataclasses import dataclass
from typing import Dict, Union

from deepracer_research.config.training.exploration_strategy import ExplorationStrategy
from deepracer_research.config.training.loss_type import LossType


@dataclass
class AWSHyperparameters:
    """AWS DeepRacer training hyperparameters."""

    batch_size: int = 64
    learning_rate: float = 0.0003
    entropy_coefficient: float = 0.01
    discount_factor: float = 0.999
    loss_type: LossType = LossType.HUBER
    num_episodes_between_training: int = 20
    num_epochs: int = 10
    stack_size: int = 1
    lr_decay_rate: float = 0.999
    beta_entropy: float = 0.01
    epsilon: float = 0.2
    exploration_type: ExplorationStrategy = ExplorationStrategy.get_default()
    e_greedy_value: float = 0.05
    epsilon_steps: int = 10000
    term_condition_max_episodes: int = 1000
    term_condition_avg_score: float = 100000

    def validate(self) -> bool:
        """Validate hyperparameters.

        Returns
        -------
        bool
            True if hyperparameters are valid
        """
        if self.batch_size <= 0:
            return False
        if not 0 < self.learning_rate < 1:
            return False
        if not 0 <= self.entropy_coefficient <= 1:
            return False
        if not 0 <= self.discount_factor <= 1:
            return False
        if self.num_episodes_between_training <= 0:
            return False
        if self.num_epochs <= 0:
            return False
        return True

    def to_dict(self) -> Dict[str, Union[int, float, str]]:
        """Convert to dictionary format.

        Returns
        -------
        Dict[str, Union[int, float, str]]
            Hyperparameters as dictionary
        """
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "entropy_coefficient": self.entropy_coefficient,
            "discount_factor": self.discount_factor,
            "loss_type": self.loss_type.value if hasattr(self.loss_type, "value") else self.loss_type,
            "num_episodes_between_training": self.num_episodes_between_training,
            "num_epochs": self.num_epochs,
            "stack_size": self.stack_size,
            "lr_decay_rate": self.lr_decay_rate,
            "beta_entropy": self.beta_entropy,
            "epsilon": self.epsilon,
            "exploration_type": self.exploration_type.value,
            "e_greedy_value": self.e_greedy_value,
            "epsilon_steps": self.epsilon_steps,
            "term_condition_max_episodes": self.term_condition_max_episodes,
            "term_condition_avg_score": self.term_condition_avg_score,
        }


DEFAULT_HYPERPARAMETERS = AWSHyperparameters()
