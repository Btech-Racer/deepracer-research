from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from deepracer_research.config.aws.aws_hyperparameters import AWSHyperparameters
from deepracer_research.config.network.architecture_type import ArchitectureType
from deepracer_research.config.training.optimizer_config import DEFAULT_DEEPRACER_OPTIMIZER, OptimizerConfig


@dataclass
class SageMakerHyperparameters(AWSHyperparameters):
    """Extended hyperparameters for SageMaker training jobs.

    Parameters
    ----------
    model_architecture : ArchitectureType
        Neural network architecture type
    hidden_size : int
        Hidden layer size for the model
    num_layers : int
        Number of layers in the model
    optimizer_config : OptimizerConfig
        Optimizer configuration with type safety
    dropout_rate : float
        Dropout rate for regularization
    batch_norm : bool
        Whether to use batch normalization
    gradient_clipping : float
        Gradient clipping threshold
    early_stopping_patience : int
        Early stopping patience in epochs
    reduce_lr_patience : int
        Learning rate reduction patience
    custom_params : Dict[str, Any]
        Additional custom parameters
    """

    model_architecture: ArchitectureType = ArchitectureType.ATTENTION_CNN
    hidden_size: int = 512
    num_layers: int = 3

    optimizer_config: OptimizerConfig = field(default_factory=lambda: DEFAULT_DEEPRACER_OPTIMIZER)

    dropout_rate: float = 0.1
    batch_norm: bool = True

    gradient_clipping: float = 1.0
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5

    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_sagemaker_format(self) -> Dict[str, str]:
        """Convert to SageMaker hyperparameter format (all values as strings).

        Returns
        -------
        Dict[str, str]
            Hyperparameters formatted for SageMaker API with type safety
        """
        aws_params = super().to_dict()

        hyperparams: Dict[str, str] = {key: str(value) for key, value in aws_params.items()}

        sagemaker_params = {
            "model_architecture": self.model_architecture,
            "hidden_size": str(self.hidden_size),
            "num_layers": str(self.num_layers),
            "dropout_rate": str(self.dropout_rate),
            "batch_norm": str(self.batch_norm),
            "gradient_clipping": str(self.gradient_clipping),
            "early_stopping_patience": str(self.early_stopping_patience),
            "reduce_lr_patience": str(self.reduce_lr_patience),
        }

        hyperparams.update(sagemaker_params)

        optimizer_params = self.optimizer_config.to_string_format()
        for key, value in optimizer_params.items():
            hyperparams[f"optimizer_{key}"] = value

        for key, value in self.custom_params.items():
            hyperparams[key] = str(value)

        return hyperparams

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format extending the AWS base parameters.

        Returns
        -------
        Dict[str, Any]
            Complete hyperparameters as dictionary
        """
        params = super().to_dict()

        sagemaker_params = {
            "model_architecture": self.model_architecture,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout_rate": self.dropout_rate,
            "batch_norm": self.batch_norm,
            "gradient_clipping": self.gradient_clipping,
            "early_stopping_patience": self.early_stopping_patience,
            "reduce_lr_patience": self.reduce_lr_patience,
            "optimizer_config": self.optimizer_config.to_dict(),
            "custom_params": self.custom_params,
        }

        params.update(sagemaker_params)
        return params

    def validate(self) -> bool:
        """Validate hyperparameters including SageMaker-specific ones.

        Returns
        -------
        bool
            True if hyperparameters are valid with type checking
        """
        if not super().validate():
            return False

        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            return False
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            return False
        if not isinstance(self.dropout_rate, (int, float)) or not 0 <= self.dropout_rate <= 1:
            return False
        if not isinstance(self.batch_norm, bool):
            return False
        if not isinstance(self.gradient_clipping, (int, float)) or self.gradient_clipping <= 0:
            return False
        if not isinstance(self.early_stopping_patience, int) or self.early_stopping_patience < 0:
            return False
        if not isinstance(self.reduce_lr_patience, int) or self.reduce_lr_patience < 0:
            return False
        if not isinstance(self.model_architecture, str) or not self.model_architecture.strip():
            return False

        if not self.optimizer_config.validate():
            return False

        if not isinstance(self.custom_params, dict):
            return False

        return True

    @classmethod
    def for_research(cls, optimizer_config: Optional[OptimizerConfig] = None, **kwargs) -> "SageMakerHyperparameters":
        """Create hyperparameters optimized for research scenarios.

        Parameters
        ----------
        optimizer_config : Optional[OptimizerConfig]
            Custom optimizer configuration
        **kwargs
            Additional parameter overrides

        Returns
        -------
        SageMakerHyperparameters
            Research-optimized configuration
        """
        if optimizer_config is None:
            optimizer_config = OptimizerConfig.for_fast_training()

        defaults = {
            "model_architecture": "advanced_cnn",
            "hidden_size": 1024,
            "num_layers": 5,
            "optimizer_config": optimizer_config,
            "dropout_rate": 0.15,
            "batch_norm": True,
            "gradient_clipping": 0.5,
            "early_stopping_patience": 15,
        }

        return cls(**{**defaults, **kwargs})

    @classmethod
    def for_production(cls, optimizer_config: Optional[OptimizerConfig] = None, **kwargs) -> "SageMakerHyperparameters":
        """Create hyperparameters optimized for production training.

        Parameters
        ----------
        optimizer_config : Optional[OptimizerConfig]
            Custom optimizer configuration
        **kwargs
            Additional parameter overrides

        Returns
        -------
        SageMakerHyperparameters
            Production-optimized configuration
        """
        if optimizer_config is None:
            optimizer_config = OptimizerConfig.for_stable_training()

        defaults = {
            "model_architecture": "efficient_cnn",
            "hidden_size": 512,
            "num_layers": 3,
            "optimizer_config": optimizer_config,
            "dropout_rate": 0.1,
            "batch_norm": True,
            "gradient_clipping": 1.0,
            "early_stopping_patience": 8,
        }

        return cls(**{**defaults, **kwargs})


DEFAULT_SAGEMAKER_HYPERPARAMETERS = SageMakerHyperparameters()
