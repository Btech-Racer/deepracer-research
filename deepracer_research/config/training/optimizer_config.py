from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class OptimizerType(str, Enum):
    """Supported optimizer types for training."""

    ADAM = "adam"
    """Adam optimizer (Kingma & Ba, 2014).

    Adaptive moment estimation with bias correction.
    Good default choice for most deep learning applications.
    """

    ADAMW = "adamw"
    """AdamW optimizer (Loshchilov & Hutter, 2017).

    Adam with decoupled weight decay regularization.
    Often better than Adam for transformer models.
    """

    SGD = "sgd"
    """Stochastic Gradient Descent.

    Classic optimizer with optional momentum.
    Good for fine-tuning and when strong regularization is needed.
    """

    RMSPROP = "rmsprop"
    """RMSprop optimizer (Hinton, 2012).

    Adaptive learning rate method that divides by running average
    of squared gradients. Good for recurrent networks.
    """

    ADAGRAD = "adagrad"
    """Adagrad optimizer (Duchi et al., 2011).

    Adaptive gradient algorithm with per-parameter learning rates.
    Good for sparse data but can have diminishing learning rates.
    """


@dataclass
class OptimizerConfig:
    """Configuration for training optimizers.

    This dataclass encapsulates optimizer settings for various training
    scenarios in DeepRacer and other machine learning applications.

    Parameters
    ----------
    optimizer_type : OptimizerType
        Type of optimizer to use
    learning_rate : float, optional
        Base learning rate, by default 3e-4
    weight_decay : float, optional
        Weight decay (L2 regularization), by default 0.01
    beta1 : float, optional
        Beta1 parameter for Adam-like optimizers, by default 0.9
    beta2 : float, optional
        Beta2 parameter for Adam-like optimizers, by default 0.999
    epsilon : float, optional
        Epsilon for numerical stability, by default 1e-8
    momentum : float, optional
        Momentum for SGD, by default 0.9
    nesterov : bool, optional
        Use Nesterov momentum for SGD, by default False
    amsgrad : bool, optional
        Use AMSGrad variant for Adam, by default False
    custom_params : Dict[str, Any], optional
        Additional optimizer-specific parameters
    """

    optimizer_type: OptimizerType = OptimizerType.ADAM
    learning_rate: float = 3e-4
    weight_decay: float = 0.01

    # Adam/AdamW parameters
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    amsgrad: bool = False

    # SGD parameters
    momentum: float = 0.9
    nesterov: bool = False

    # RMSprop parameters
    rho: float = 0.9

    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def to_tensorflow_config(self) -> Dict[str, Any]:
        """Convert to TensorFlow optimizer configuration.

        Returns
        -------
        Dict[str, Any]
            TensorFlow optimizer configuration
        """
        base_config = {"learning_rate": self.learning_rate, **self.custom_params}

        if self.optimizer_type == OptimizerType.ADAM:
            return {
                "optimizer": "adam",
                **base_config,
                "beta_1": self.beta1,
                "beta_2": self.beta2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        elif self.optimizer_type == OptimizerType.ADAMW:
            return {
                "optimizer": "adamw",
                **base_config,
                "weight_decay": self.weight_decay,
                "beta_1": self.beta1,
                "beta_2": self.beta2,
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
            }
        elif self.optimizer_type == OptimizerType.SGD:
            return {"optimizer": "sgd", **base_config, "momentum": self.momentum, "nesterov": self.nesterov}
        elif self.optimizer_type == OptimizerType.RMSPROP:
            return {"optimizer": "rmsprop", **base_config, "rho": self.rho, "momentum": self.momentum, "epsilon": self.epsilon}
        elif self.optimizer_type == OptimizerType.ADAGRAD:
            return {"optimizer": "adagrad", **base_config, "epsilon": self.epsilon}
        else:
            return base_config

    def to_pytorch_config(self) -> Dict[str, Any]:
        """Convert to PyTorch optimizer configuration.

        Returns
        -------
        Dict[str, Any]
            PyTorch optimizer configuration
        """
        base_config = {"lr": self.learning_rate, **self.custom_params}

        if self.optimizer_type == OptimizerType.ADAM:
            return {
                "optimizer": "Adam",
                **base_config,
                "betas": (self.beta1, self.beta2),
                "eps": self.epsilon,
                "weight_decay": self.weight_decay,
                "amsgrad": self.amsgrad,
            }
        elif self.optimizer_type == OptimizerType.ADAMW:
            return {
                "optimizer": "AdamW",
                **base_config,
                "betas": (self.beta1, self.beta2),
                "eps": self.epsilon,
                "weight_decay": self.weight_decay,
                "amsgrad": self.amsgrad,
            }
        elif self.optimizer_type == OptimizerType.SGD:
            return {
                "optimizer": "SGD",
                **base_config,
                "momentum": self.momentum,
                "weight_decay": self.weight_decay,
                "nesterov": self.nesterov,
            }
        elif self.optimizer_type == OptimizerType.RMSPROP:
            return {
                "optimizer": "RMSprop",
                **base_config,
                "alpha": self.rho,
                "eps": self.epsilon,
                "weight_decay": self.weight_decay,
                "momentum": self.momentum,
            }
        elif self.optimizer_type == OptimizerType.ADAGRAD:
            return {"optimizer": "Adagrad", **base_config, "eps": self.epsilon, "weight_decay": self.weight_decay}
        else:
            return base_config

    def to_string_format(self) -> Dict[str, str]:
        """Convert all parameters to string format (useful for API calls).

        Returns
        -------
        Dict[str, str]
            All parameters as strings
        """
        config = {
            "optimizer_type": self.optimizer_type.value,
            "learning_rate": str(self.learning_rate),
            "weight_decay": str(self.weight_decay),
            "beta1": str(self.beta1),
            "beta2": str(self.beta2),
            "epsilon": str(self.epsilon),
            "momentum": str(self.momentum),
            "nesterov": str(self.nesterov),
            "amsgrad": str(self.amsgrad),
            "rho": str(self.rho),
        }

        # Add custom parameters as strings
        for key, value in self.custom_params.items():
            config[key] = str(value)

        return config

    @classmethod
    def for_deepracer(cls) -> "OptimizerConfig":
        """Create optimizer config optimized for DeepRacer training.

        Returns
        -------
        OptimizerConfig
            Optimized configuration for DeepRacer
        """
        return cls(
            optimizer_type=OptimizerType.ADAM, learning_rate=3e-4, weight_decay=0.01, beta1=0.9, beta2=0.999, epsilon=1e-7
        )

    @classmethod
    def for_fast_training(cls) -> "OptimizerConfig":
        """Create optimizer config for fast training convergence.

        Returns
        -------
        OptimizerConfig
            Configuration optimized for fast convergence
        """
        return cls(optimizer_type=OptimizerType.ADAMW, learning_rate=1e-3, weight_decay=0.05, beta1=0.9, beta2=0.95)

    @classmethod
    def for_stable_training(cls) -> "OptimizerConfig":
        """Create optimizer config for stable, conservative training.

        Returns
        -------
        OptimizerConfig
            Configuration optimized for stability
        """
        return cls(optimizer_type=OptimizerType.SGD, learning_rate=1e-4, momentum=0.95, weight_decay=0.001, nesterov=True)

    def validate(self) -> bool:
        """Validate the optimizer configuration.

        Returns
        -------
        bool
            True if configuration is valid
        """
        if self.learning_rate <= 0:
            return False
        if self.weight_decay < 0:
            return False
        if not 0 <= self.beta1 <= 1:
            return False
        if not 0 <= self.beta2 <= 1:
            return False
        if self.epsilon <= 0:
            return False
        if not 0 <= self.momentum <= 1:
            return False
        if not 0 <= self.rho <= 1:
            return False

        return True


# Default configurations for common scenarios
DEFAULT_DEEPRACER_OPTIMIZER = OptimizerConfig.for_deepracer()
FAST_TRAINING_OPTIMIZER = OptimizerConfig.for_fast_training()
STABLE_TRAINING_OPTIMIZER = OptimizerConfig.for_stable_training()
