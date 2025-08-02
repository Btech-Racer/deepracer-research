from enum import Enum, unique


@unique
class NeuralNetworkType(str, Enum):
    """Neural network architectures supported by AWS DeepRacer.

    Reference: https://blog.gofynd.com/how-we-broke-into-the-top-1-of-the-aws-deepracer-virtual-circuit-573ba46c275
    """

    SHALLOW = "DEEP_CONVOLUTIONAL_NETWORK_SHALLOW"
    """3-layer Convolutional Neural Network (Recommended).

    Fast to train, efficient inference, and proven effective for time trials.
    Preferred choice for AWS DeepRacer competitions and most racing scenarios.
    Top 1% models typically use this architecture for optimal performance.
    """

    DEEP = "DEEP_CONVOLUTIONAL_NETWORK_DEEP"
    """5-layer Convolutional Neural Network (Advanced).

    More complex architecture with deeper feature extraction.
    Higher computational cost but potentially better for complex scenarios.
    Use only when shallow network proves insufficient for specific use cases.
    """

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
        layer_counts = {self.SHALLOW: 3, self.DEEP: 5}
        return layer_counts[self]

    def get_description(self) -> str:
        """Get  description of the neural network type.

        Returns
        -------
        str
            Description of the neural network architecture
        """
        descriptions = {
            self.SHALLOW: "3-layer CNN - fast training, proven performance, recommended for most use cases",
            self.DEEP: "5-layer CNN - complex feature extraction, higher computational cost, advanced scenarios only",
        }
        return descriptions[self]

    def get_training_time_multiplier(self) -> float:
        """Get approximate training time multiplier compared to shallow network.

        Returns
        -------
        float
            Training time multiplier (1.0 = baseline shallow network)
        """
        multipliers = {self.SHALLOW: 1.0, self.DEEP: 2.5}  # Significantly slower training
        return multipliers[self]

    def is_recommended_for_competitions(self) -> bool:
        """Check if this architecture is recommended for competitions.

        Returns
        -------
        bool
            True if recommended for competitive racing
        """
        return self == self.SHALLOW
