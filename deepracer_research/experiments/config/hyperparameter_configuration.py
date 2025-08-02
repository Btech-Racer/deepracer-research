from dataclasses import dataclass, field
from typing import Any, Dict, List

from deepracer_research.config.training.exploration_strategy import ExplorationStrategy
from deepracer_research.config.training.loss_type import LossType


@dataclass
class HyperparameterConfiguration:
    """Comprehensive hyperparameter configuration for experimental training."""

    batch_size: int = 64
    learning_rate: float = 0.0003
    entropy_coefficient: float = 0.01
    discount_factor: float = 0.999
    loss_type: LossType = LossType.HUBER
    num_episodes_between_training: int = 20
    num_epochs: int = 10
    stack_size: int = 1
    e_greedy_value: float = 1.0
    epsilon_steps: int = 10000
    exploration_type: ExplorationStrategy = ExplorationStrategy.CATEGORICAL
    term_condition_avg_score: float = 350000.0
    term_condition_max_episodes: int = 1000
    lr_decay_epochs: List[int] = field(default_factory=lambda: [5, 10])

    def to_aws_hyperparameters(self):
        """Convert to AWS hyperparameters format."""
        from deepracer_research.config.aws.aws_hyperparameters import AWSHyperparameters

        return AWSHyperparameters(
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            entropy_coefficient=self.entropy_coefficient,
            discount_factor=self.discount_factor,
            loss_type=self.loss_type,
            num_episodes_between_training=self.num_episodes_between_training,
            num_epochs=self.num_epochs,
            stack_size=self.stack_size,
            e_greedy_value=self.e_greedy_value,
            epsilon_steps=self.epsilon_steps,
            exploration_type=self.exploration_type,
            term_condition_avg_score=self.term_condition_avg_score,
            term_condition_max_episodes=self.term_condition_max_episodes,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "entropy_coefficient": self.entropy_coefficient,
            "discount_factor": self.discount_factor,
            "loss_type": self.loss_type.value,
            "num_episodes_between_training": self.num_episodes_between_training,
            "num_epochs": self.num_epochs,
            "stack_size": self.stack_size,
            "e_greedy_value": self.e_greedy_value,
            "epsilon_steps": self.epsilon_steps,
            "exploration_type": self.exploration_type.value,
            "term_condition_avg_score": self.term_condition_avg_score,
            "term_condition_max_episodes": self.term_condition_max_episodes,
            "lr_decay_epochs": self.lr_decay_epochs,
        }
