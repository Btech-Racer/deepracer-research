from enum import Enum, unique


@unique
class RewardFunctionType(str, Enum):
    """Types of reward functions available for AWS DeepRacer training"""

    CENTERLINE_FOLLOWING = "centerline_following"
    """Basic centerline following reward function.

    Encourages the car to stay close to the center of the track.
    Uses distance markers to provide graduated rewards.
    """

    SPEED_OPTIMIZATION = "speed_optimization"
    """Speed-focused reward function with centerline tolerance.

    Rewards high speed while maintaining reasonable track position.
    Balances speed and control for competitive racing.
    """

    OBJECT_AVOIDANCE = "object_avoidance"
    """Object avoidance reward function for obstacle scenarios.

    Rewards safe navigation around static and dynamic objects
    while maintaining progress and smooth control.
    """

    MULTI_OBJECTIVE = "multi_objective"
    """Multi-objective reward combining multiple goals.

    Balances multiple objectives like speed, safety, and efficiency
    using weighted combinations of individual reward components.
    """

    PROGRESSIVE_SPEED = "progressive_speed"
    """Progressive speed increase reward function.

    Gradually increases speed requirements as training progresses,
    starting with safety and adding speed objectives over time.
    """

    WAYPOINT_FOLLOWING = "waypoint_following"
    """Waypoint-based navigation reward function.

    Rewards following specific waypoints with optimal racing lines
    and trajectory planning for track-specific optimization.
    """

    CUSTOM = "custom"
    """Custom user-defined reward function.

    Placeholder for custom reward functions implemented by users
    for specific research or application requirements.
    """

    DEFAULT = "default"
    """Default AWS DeepRacer reward function.

    Standard reward function provided by AWS DeepRacer service,
    typically focused on basic centerline following.
    """

    @classmethod
    def get_template_compatible_types(cls) -> list:
        """Get reward function types that have YAML templates.

        Returns
        -------
        list
            List of RewardFunctionType values that have corresponding templates
        """
        return [cls.CENTERLINE_FOLLOWING, cls.SPEED_OPTIMIZATION, cls.OBJECT_AVOIDANCE]

    @classmethod
    def get_scenario_compatible_types(cls) -> list:
        """Get reward function types compatible with scenario-based training.

        Returns
        -------
        list
            List of RewardFunctionType values suitable for scenario training
        """
        return [
            cls.CENTERLINE_FOLLOWING,
            cls.SPEED_OPTIMIZATION,
            cls.OBJECT_AVOIDANCE,
            cls.MULTI_OBJECTIVE,
            cls.WAYPOINT_FOLLOWING,
        ]

    def get_description(self) -> str:
        """Description of the reward function type.

        Returns
        -------
        str
            Description of the reward function type
        """
        descriptions = {
            self.CENTERLINE_FOLLOWING: "Basic centerline following with distance-based rewards",
            self.SPEED_OPTIMIZATION: "High-speed racing with centerline tolerance",
            self.OBJECT_AVOIDANCE: "Safe navigation around obstacles",
            self.MULTI_OBJECTIVE: "Balanced multiple objectives",
            self.PROGRESSIVE_SPEED: "Gradually increasing speed requirements",
            self.WAYPOINT_FOLLOWING: "Optimal racing line via waypoint navigation",
            self.CUSTOM: "User-defined custom reward function",
            self.DEFAULT: "Standard AWS DeepRacer reward function",
        }
        return descriptions.get(self, "Unknown reward function type")

    def is_template_based(self) -> bool:
        """Check if this reward function type uses YAML templates.

        Returns
        -------
        bool
            True if the reward function uses YAML templates, False otherwise
        """
        return self in self.get_template_compatible_types()
