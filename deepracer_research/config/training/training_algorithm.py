from enum import Enum, unique


@unique
class TrainingAlgorithm(str, Enum):
    """Core training algorithms for AWS DeepRacer and autonomous racing research.

    Focused on the most essential and practical algorithms for racing applications,
    covering different paradigms and use cases.
    """

    PPO = "ppo"
    """Proximal Policy Optimization (Schulman et al., 2017).

    AWS DeepRacer's default algorithm. Stable, sample-efficient, and robust.
    Excellent balance of performance and reliability for racing applications.
    """

    CLIPPED_PPO = "clipped_ppo"
    """Clipped Proximal Policy Optimization (Intel Labs Coach).

    AWS DeepRacer's primary implementation with likelihood ratio clipping for stability.
    Uses unified loss function with Adam optimizer and GAE-based value targets.
    Proven effective for top 1% DeepRacer performance (Fynd Engineering, 2020).
    Reference: https://intellabs.github.io/coach/components/agents/policy_optimization/cppo.html
    """

    SAC = "sac"
    """Soft Actor-Critic (Haarnoja et al., 2018).

    Maximum entropy RL algorithm excellent for continuous control in racing.
    Superior exploration and sample efficiency compared to PPO.
    """

    TD3 = "td3"
    """Twin Delayed Deep Deterministic Policy Gradient (Fujimoto et al., 2018).

    Advanced off-policy algorithm with improved stability over DDPG.
    Excellent for precise steering and speed control in racing.
    """

    RAINBOW_DQN = "rainbow_dqn"
    """Rainbow DQN (Hessel et al., 2018).

    State-of-the-art value-based method combining six DQN improvements.
    Best choice for discrete action spaces in racing scenarios.
    """

    MAML = "maml"
    """Model-Agnostic Meta-Learning (Finn et al., 2017).

    Meta-learning algorithm for rapid adaptation to new tracks and conditions.
    Essential for research in adaptive racing systems.
    """

    MADDPG = "maddpg"
    """Multi-Agent Deep Deterministic Policy Gradient (Lowe et al., 2017).

    Multi-agent algorithm for head-to-head racing and competitive scenarios.
    Handles non-stationary environments with multiple racing agents.
    """

    DREAMER_V2 = "dreamer_v2"
    """Mastering Atari with Discrete World Models (Hafner et al., 2020).

    Model-based RL algorithm that learns world models for planning.
    Highly sample efficient for complex racing scenarios.
    """

    @classmethod
    def get_aws_supported(cls) -> list:
        """Get algorithms officially supported by AWS DeepRacer."""
        return [cls.PPO, cls.CLIPPED_PPO, cls.SAC]

    @classmethod
    def get_continuous_control(cls) -> list:
        """Get algorithms best for continuous steering/throttle control."""
        return [cls.SAC, cls.TD3, cls.PPO, cls.CLIPPED_PPO]

    @classmethod
    def get_discrete_control(cls) -> list:
        """Get algorithms best for discrete action spaces."""
        return [cls.PPO, cls.CLIPPED_PPO, cls.RAINBOW_DQN]

    @classmethod
    def get_multi_agent(cls) -> list:
        """Get algorithms for multi-agent racing scenarios."""
        return [cls.MADDPG]

    @classmethod
    def get_research_algorithms(cls) -> list:
        """Get advanced algorithms for research applications."""
        return [cls.TD3, cls.RAINBOW_DQN, cls.MAML, cls.DREAMER_V2]

    def get_description(self) -> str:
        """Get concise description of the algorithm."""
        descriptions = {
            self.PPO: "AWS DeepRacer default - stable, reliable, good starting point",
            self.CLIPPED_PPO: "AWS DeepRacer production - policy clipping for stability, proven top 1% performance",
            self.SAC: "Maximum entropy RL - best for continuous control and exploration",
            self.TD3: "Advanced off-policy - precise control with improved stability",
            self.RAINBOW_DQN: "State-of-the-art value-based - best for discrete actions",
            self.MAML: "Meta-learning - rapid adaptation to new racing conditions",
            self.MADDPG: "Multi-agent RL - head-to-head and competitive racing",
            self.DREAMER_V2: "Model-based RL - sample efficient with world models",
        }
        return descriptions[self]

    def get_use_case(self) -> str:
        """Get primary use case for this algorithm."""
        use_cases = {
            self.PPO: "General racing, beginner-friendly, stable training",
            self.CLIPPED_PPO: "Competitive racing, time trials, leader board optimization",
            self.SAC: "Continuous control, complex tracks, maximum performance",
            self.TD3: "Precise control, deterministic policies, stability focus",
            self.RAINBOW_DQN: "Discrete actions, distributional learning, uncertainty",
            self.MAML: "Track adaptation, few-shot learning, varying conditions",
            self.MADDPG: "Multi-car racing, competitive scenarios, head-to-head",
            self.DREAMER_V2: "Sample efficiency, world modeling, complex environments",
        }
        return use_cases[self]

    def get_reference(self) -> str:
        """Get reference for the algorithm."""
        references = {
            self.PPO: "Schulman, J., et al. (2017). Proximal Policy Optimization Algorithms",
            self.CLIPPED_PPO: "Intel Labs Coach (2019). Clipped Proximal Policy Optimization; Fynd Engineering (2020). Top 1% AWS DeepRacer Virtual Circuit",
            self.SAC: "Haarnoja, T., et al. (2018). Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL",
            self.TD3: "Fujimoto, S., et al. (2018). Addressing Function Approximation Error in Actor-Critic Methods",
            self.RAINBOW_DQN: "Hessel, M., et al. (2018). Rainbow: Combining Improvements in Deep Reinforcement Learning",
            self.MAML: "Finn, C., et al. (2017). Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
            self.MADDPG: "Lowe, R., et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments",
            self.DREAMER_V2: "Hafner, D., et al. (2020). Mastering Atari with Discrete World Models",
        }
        return references[self]
