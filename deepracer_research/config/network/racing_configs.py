from typing import List

from deepracer_research.config.network.architecture_type import ArchitectureType
from deepracer_research.config.network.network_config import NetworkConfig

RACING_CONFIGS = {
    "attention_cnn_racing": NetworkConfig(
        architecture_type=ArchitectureType.ATTENTION_CNN,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[256, 128, 64],
        dropout_rate=0.1,
        learning_rate=1e-4,
    ),
    "obstacle_avoidance": NetworkConfig(
        architecture_type=ArchitectureType.ATTENTION_CNN,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[256, 128],
        dropout_rate=0.15,
        learning_rate=3e-4,
    ),
    "continuous_control": NetworkConfig(
        architecture_type=ArchitectureType.RESIDUAL_NETWORK,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2,
        learning_rate=5e-4,
    ),
    "efficient_racing": NetworkConfig(
        architecture_type=ArchitectureType.EFFICIENT_NET,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[256, 128],
        dropout_rate=0.1,
        learning_rate=1e-4,
    ),
    "high_performance": NetworkConfig(
        architecture_type=ArchitectureType.RESIDUAL_NETWORK,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.15,
        learning_rate=3e-4,
    ),
    "temporal_racing": NetworkConfig(
        architecture_type=ArchitectureType.TEMPORAL_CNN,
        input_shape=(5, 160, 120, 3),
        num_actions=2,
        hidden_dims=[256, 128],
        dropout_rate=0.2,
        learning_rate=2e-4,
    ),
    "transformer_racing": NetworkConfig(
        architecture_type=ArchitectureType.TRANSFORMER_VISION,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[512, 256],
        dropout_rate=0.1,
        learning_rate=1e-4,
    ),
    "multi_modal_racing": NetworkConfig(
        architecture_type=ArchitectureType.MULTI_MODAL_FUSION,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.15,
        learning_rate=2e-4,
    ),
    "sequential_racing": NetworkConfig(
        architecture_type=ArchitectureType.LSTM_CNN,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[256, 128],
        dropout_rate=0.2,
        learning_rate=3e-4,
    ),
    "speed_optimized_attention": NetworkConfig(
        architecture_type=ArchitectureType.ATTENTION_CNN,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[128, 64],
        dropout_rate=0.05,
        learning_rate=2e-4,
    ),
    "speed_optimized_efficient": NetworkConfig(
        architecture_type=ArchitectureType.EFFICIENT_NET,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[128, 64],
        dropout_rate=0.05,
        learning_rate=1e-4,
    ),
    "precision_residual": NetworkConfig(
        architecture_type=ArchitectureType.RESIDUAL_NETWORK,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[512, 256, 128, 64],
        dropout_rate=0.25,
        learning_rate=1e-4,
    ),
    "precision_transformer": NetworkConfig(
        architecture_type=ArchitectureType.TRANSFORMER_VISION,
        input_shape=(160, 120, 3),
        num_actions=2,
        hidden_dims=[512, 256, 128],
        dropout_rate=0.2,
        learning_rate=5e-5,
    ),
}

SCENARIO_CONFIGS = {
    "centerline_following": "attention_cnn_racing",
    "speed_optimization": "speed_optimized_efficient",
    "obstacle_avoidance": "obstacle_avoidance",
    "time_trial": "speed_optimized_attention",
    "head_to_head": "high_performance",
    "multi_agent": "precision_transformer",
    "continuous_control": "continuous_control",
}

ARCHITECTURE_CONFIGS = {
    ArchitectureType.ATTENTION_CNN: "attention_cnn_racing",
    ArchitectureType.RESIDUAL_NETWORK: "high_performance",
    ArchitectureType.EFFICIENT_NET: "efficient_racing",
    ArchitectureType.TEMPORAL_CNN: "temporal_racing",
    ArchitectureType.TRANSFORMER_VISION: "transformer_racing",
    ArchitectureType.MULTI_MODAL_FUSION: "multi_modal_racing",
    ArchitectureType.LSTM_CNN: "sequential_racing",
}


class RacingConfigManager:
    """Manager class for racing configurations with parameter overrides."""

    def __init__(self):
        self._configs = RACING_CONFIGS.copy()
        self._scenario_configs = SCENARIO_CONFIGS.copy()
        self._architecture_configs = ARCHITECTURE_CONFIGS.copy()

    def get_config(self, config_name: str, **overrides) -> NetworkConfig:
        """Get a racing configuration with optional parameter overrides.

        Parameters
        ----------
        config_name : str
            Name of the racing configuration
        **overrides
            Optional parameter overrides (e.g., learning_rate=1e-3, dropout_rate=0.2)

        Returns
        -------
        NetworkConfig
            Racing configuration with applied overrides

        Raises
        ------
        KeyError
            If configuration name is not found
        ValueError
            If override parameter is invalid
        """
        if config_name not in self._configs:
            available = list(self._configs.keys())
            raise KeyError(f"Configuration '{config_name}' not found. Available: {available}")

        base_config = self._configs[config_name]

        if not overrides:
            return base_config

        config_dict = {
            "architecture_type": base_config.architecture_type,
            "input_shape": base_config.input_shape,
            "num_actions": base_config.num_actions,
            "hidden_dims": base_config.hidden_dims,
            "attention_heads": base_config.attention_heads,
            "dropout_rate": base_config.dropout_rate,
            "use_batch_norm": base_config.use_batch_norm,
            "activation": base_config.activation,
            "output_activation": base_config.output_activation,
            "learning_rate": base_config.learning_rate,
        }

        for key, value in overrides.items():
            if key not in config_dict:
                valid_keys = list(config_dict.keys())
                raise ValueError(f"Invalid override parameter '{key}'. Valid parameters: {valid_keys}")
            config_dict[key] = value

        return NetworkConfig(**config_dict)

    def get_config_for_scenario(self, scenario_name: str, **overrides) -> NetworkConfig:
        """Get configuration optimized for a racing scenario with overrides.

        Parameters
        ----------
        scenario_name : str
            Name of the racing scenario
        **overrides
            Optional parameter overrides

        Returns
        -------
        NetworkConfig
            Optimized network configuration with overrides
        """
        if scenario_name not in self._scenario_configs:
            available = list(self._scenario_configs.keys())
            raise KeyError(f"Scenario '{scenario_name}' not found. Available: {available}")

        config_name = self._scenario_configs[scenario_name]
        return self.get_config(config_name, **overrides)

    def get_config_for_architecture(self, architecture: ArchitectureType, **overrides) -> NetworkConfig:
        """Get configuration optimized for an architecture with overrides.

        Parameters
        ----------
        architecture : ArchitectureType
            The architecture type
        **overrides
            Optional parameter overrides

        Returns
        -------
        NetworkConfig
            Optimized network configuration with overrides
        """
        if architecture not in self._architecture_configs:
            available = list(self._architecture_configs.keys())
            raise KeyError(f"Architecture '{architecture}' not found. Available: {available}")

        config_name = self._architecture_configs[architecture]
        return self.get_config(config_name, **overrides)

    def list_available_configs(self) -> List[str]:
        """List all available racing configurations."""
        return list(self._configs.keys())

    def list_supported_scenarios(self) -> List[str]:
        """List all supported racing scenarios."""
        return list(self._scenario_configs.keys())

    def list_supported_architectures(self) -> List[ArchitectureType]:
        """List all supported architecture types."""
        return list(self._architecture_configs.keys())

    def add_custom_config(self, name: str, config: NetworkConfig) -> None:
        """Add a custom racing configuration.

        Parameters
        ----------
        name : str
            Name for the custom configuration
        config : NetworkConfig
            The network configuration
        """
        self._configs[name] = config

    def remove_config(self, name: str) -> None:
        """Remove a racing configuration.

        Parameters
        ----------
        name : str
            Name of the configuration to remove
        """
        if name in self._configs:
            del self._configs[name]


racing_config_manager = RacingConfigManager()


def get_config_for_scenario(scenario_name: str) -> NetworkConfig:
    """Get optimized configuration for a racing scenario.

    Parameters
    ----------
    scenario_name : str
        Name of the racing scenario

    Returns
    -------
    NetworkConfig
        Optimized network configuration

    Raises
    ------
    KeyError
        If scenario is not supported
    """
    config_name = SCENARIO_CONFIGS.get(scenario_name)
    if not config_name:
        raise KeyError(f"No configuration found for scenario: {scenario_name}")

    return RACING_CONFIGS[config_name]


def get_config_for_architecture(architecture: ArchitectureType) -> NetworkConfig:
    """Get optimized configuration for an architecture type.

    Parameters
    ----------
    architecture : ArchitectureType
        The architecture type

    Returns
    -------
    NetworkConfig
        Optimized network configuration

    Raises
    ------
    KeyError
        If architecture is not supported
    """
    config_name = ARCHITECTURE_CONFIGS.get(architecture)
    if not config_name:
        raise KeyError(f"No configuration found for architecture: {architecture}")

    return RACING_CONFIGS[config_name]


def list_available_configs() -> list:
    """List all available racing configurations.

    Returns
    -------
    list
        List of available configuration names
    """
    return list(RACING_CONFIGS.keys())


def list_supported_scenarios() -> list:
    """List all supported racing scenarios.

    Returns
    -------
    list
        List of supported scenario names
    """
    return list(SCENARIO_CONFIGS.keys())


def list_supported_architectures() -> list:
    """List all supported architecture types.

    Returns
    -------
    list
        List of supported architecture types
    """
    return list(ARCHITECTURE_CONFIGS.keys())
