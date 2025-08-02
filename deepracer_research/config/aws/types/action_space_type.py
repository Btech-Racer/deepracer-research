from enum import Enum, unique


@unique
class ActionSpaceType(str, Enum):
    """Type of action space for AWS DeepRacer"""

    CONTINUOUS = "continuous"
    """Continuous action space with real-valued speed and steering outputs.

    Allows for smooth, analog-like control with infinite precision within
    defined ranges. Better for fine-grained control but more complex to train.
    Suitable for advanced scenarios requiring precise maneuvering.
    """

    DISCRETE = "discrete"
    """Discrete action space with predefined speed and steering combinations.

    Uses a finite set of action combinations (e.g., 15 actions = 3 speeds × 5 steering angles).
    Easier to train and debug, more stable convergence. Recommended for most
    DeepRacer scenarios including time trials and competitions.
    """

    @classmethod
    def get_recommended(cls) -> "ActionSpaceType":
        """Get the recommended action space type for most use cases.

        Returns
        -------
        ActionSpaceType
            DISCRETE - recommended for stability and ease of training
        """
        return cls.DISCRETE

    @classmethod
    def get_for_precision_control(cls) -> "ActionSpaceType":
        """Get action space type for scenarios requiring precise control.

        Returns
        -------
        ActionSpaceType
            CONTINUOUS - for scenarios needing fine-grained control
        """
        return cls.CONTINUOUS

    def is_continuous(self) -> bool:
        """Check if this is a continuous action space.

        Returns
        -------
        bool
            True if continuous action space
        """
        return self == self.CONTINUOUS

    def is_discrete(self) -> bool:
        """Check if this is a discrete action space.

        Returns
        -------
        bool
            True if discrete action space
        """
        return self == self.DISCRETE

    def get_description(self) -> str:
        """Get description of the action space type.

        Returns
        -------
        str
            Description of the action space type
        """
        descriptions = {
            self.CONTINUOUS: "Continuous action space - real-valued control, precise but complex",
            self.DISCRETE: "Discrete action space - predefined actions, stable and recommended",
        }
        return descriptions[self]

    def get_training_complexity(self) -> str:
        """Get the training complexity level for this action space type.

        Returns
        -------
        str
            Training complexity level
        """
        complexity = {
            self.CONTINUOUS: "Advanced - requires careful tuning",
            self.DISCRETE: "Beginner-friendly - stable convergence",
        }
        return complexity[self]
