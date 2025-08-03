from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from deepracer_research.config.network.activation_type import ActivationType
from deepracer_research.config.network.architecture_type import ArchitectureType


@dataclass
class NetworkConfig:
    """Configuration for advanced neural network architectures."""

    architecture_type: ArchitectureType
    input_shape: Tuple[int, ...]
    num_actions: int
    hidden_dims: Optional[List[int]] = field(default=None)
    attention_heads: int = field(default=8)
    dropout_rate: float = field(default=0.1)
    use_batch_norm: bool = field(default=True)
    activation: ActivationType = field(default=ActivationType.RELU)
    output_activation: ActivationType = field(default=ActivationType.LINEAR)
    learning_rate: float = field(default=3e-4)

    @property
    def output_size(self) -> int:
        """Get the output size (same as num_actions for compatibility)."""
        return self.num_actions

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary with string values for serialization.

        This method ensures that all enum values are converted to strings
        to avoid TensorFlow _DictWrapper serialization issues.

        Returns
        -------
        Dict[str, Any]
            Configuration as dictionary with serializable values
        """
        return {
            "architecture_type": (
                self.architecture_type.value if hasattr(self.architecture_type, "value") else str(self.architecture_type)
            ),
            "input_shape": self.input_shape,
            "num_actions": self.num_actions,
            "hidden_dims": self.hidden_dims,
            "attention_heads": self.attention_heads,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "activation": self.activation.value if hasattr(self.activation, "value") else str(self.activation),
            "output_activation": (
                self.output_activation.value if hasattr(self.output_activation, "value") else str(self.output_activation)
            ),
            "learning_rate": self.learning_rate,
            "output_size": self.output_size,
        }

    def validate(self) -> bool:
        """Validate network configuration parameters."""
        from deepracer_research.utils.logger import warning

        if not isinstance(self.input_shape, tuple) or len(self.input_shape) < 2:
            warning(f"Invalid input_shape: must be tuple with at least 2 dimensions, got {self.input_shape}")
            return False
        if self.num_actions <= 0:
            warning(f"Invalid num_actions: must be positive, got {self.num_actions}")
            return False
        if not 0.0 <= self.dropout_rate <= 1.0:
            warning(f"Invalid dropout_rate: must be between 0.0 and 1.0, got {self.dropout_rate}")
            return False
        if self.attention_heads <= 0:
            warning(f"Invalid attention_heads: must be positive, got {self.attention_heads}")
            return False
        if not isinstance(self.activation, ActivationType):
            warning(f"Invalid activation: must be ActivationType enum, got {type(self.activation)}")
            return False
        if not isinstance(self.output_activation, ActivationType):
            warning(f"Invalid output_activation: must be ActivationType enum, got {type(self.output_activation)}")
            return False
        return True
