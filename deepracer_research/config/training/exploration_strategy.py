from enum import Enum


class ExplorationStrategy(str, Enum):
    """Exploration strategies for reinforcement learning in AWS DeepRacer.

    Defines how the RL agent handles the exploration-exploitation trade-off during training.
    """

    CATEGORICAL = "categorical"
    """
    Categorical Exploration (Default)

    Used with discrete action spaces where the agent chooses from a predefined set of actions.
    The neural network outputs a probability distribution over actions and samples from it.
    Even if one action has high probability (exploitation), there's still chance for other
    actions to be selected (exploration). The exploration level is controlled by entropy.
    Higher entropy = more uncertainty and exploration.
    """

    EPSILON_GREEDY = "epsilon-greedy"
    """
    Epsilon-Greedy Exploration

    Simpler approach where epsilon (ε) value between 0-1 controls exploration:
    - With probability ε: choose random action (exploration)
    - With probability (1-ε): choose best known action (exploitation)
    Epsilon often decays over time as agent becomes more confident.
    """

    @classmethod
    def get_default(cls) -> "ExplorationStrategy":
        """Get the default exploration strategy.

        Returns
        -------
        ExplorationStrategy
            Default strategy (CATEGORICAL)
        """
        return cls.CATEGORICAL

    def is_categorical(self) -> bool:
        """Check if this is categorical exploration.

        Returns
        -------
        bool
            True if categorical exploration
        """
        return self == self.CATEGORICAL

    def is_epsilon_greedy(self) -> bool:
        """Check if this is epsilon-greedy exploration.

        Returns
        -------
        bool
            True if epsilon-greedy exploration
        """
        return self == self.EPSILON_GREEDY

    def get_description(self) -> str:
        """Get  description of the exploration strategy.

        Returns
        -------
        str
            Description of the exploration strategy
        """
        descriptions = {
            self.CATEGORICAL: "Categorical exploration with probability distribution sampling",
            self.EPSILON_GREEDY: "Epsilon-greedy exploration with random vs best action selection",
        }
        return descriptions.get(self, "Unknown exploration strategy")
