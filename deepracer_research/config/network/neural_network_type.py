from enum import Enum, unique


@unique
class NeuralNetworkType(str, Enum):
    """Neural network architectures supported by AWS DeepRacer."""

    SHALLOW = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"

    STANDARD = "DEEP_CONVOLUTIONAL_NETWORK"  # Deprecated alias for SHALLOW

    DEEP = "DEEP_CONVOLUTIONAL_NETWORK_DEEP"

    @classmethod
    def get_recommended(cls) -> "NeuralNetworkType":
        """Get the recommended neural network type for most use cases.

        Returns
        -------
        NeuralNetworkType
            SHALLOW - recommended for most racing scenarios
        """
        return cls.SHALLOW

    @classmethod
    def get_for_time_trials(cls) -> "NeuralNetworkType":
        """Get the optimal neural network type for time trial competitions.

        Returns
        -------
        NeuralNetworkType
            SHALLOW - proven effective for time trials
        """
        return cls.SHALLOW

    @classmethod
    def get_for_complex_scenarios(cls) -> "NeuralNetworkType":
        """Get neural network type for complex multi-agent scenarios.

        Returns
        -------
        NeuralNetworkType
            DEEP - for complex scenarios requiring deeper feature extraction
        """
        return cls.DEEP

    def get_layer_count(self) -> int:
        """Get the number of convolutional layers.

        Returns
        -------
        int
            Number of CNN layers in the architecture
        """
        layer_counts = {self.SHALLOW: 0, self.DEEP: 2}
        return layer_counts[self]

    def get_training_time_multiplier(self) -> float:
        """Get approximate training time multiplier compared to shallow network.

        Returns
        -------
        float
            Training time multiplier
        """
        multipliers = {self.SHALLOW: 1.0, self.DEEP: 2.5}
        return multipliers[self]

    def is_recommended_for_competitions(self) -> bool:
        """Check if this architecture is recommended for competitions.

        Returns
        -------
        bool
            True if recommended for competitive racing
        """
        return self == self.SHALLOW
