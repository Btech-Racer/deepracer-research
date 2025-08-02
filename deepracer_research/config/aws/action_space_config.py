from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from deepracer_research.config.aws.types.action_space_type import ActionSpaceType


@dataclass
class ActionSpaceConfig:
    """Unified configuration for AWS DeepRacer action space."""

    type: ActionSpaceType = ActionSpaceType.CONTINUOUS
    speed_range: Optional[Dict[str, float]] = None
    steering_range: Optional[Dict[str, float]] = None
    discrete_actions: Optional[Dict[str, Any]] = None

    num_speed_levels: int = 3
    num_steering_levels: int = 5

    def __post_init__(self):
        """Initialize default ranges based on action space type."""
        if self.type == ActionSpaceType.CONTINUOUS and self.speed_range is None:
            self.speed_range = {"min": 0.5, "max": 4.0}

        if self.type == ActionSpaceType.CONTINUOUS and self.steering_range is None:
            self.steering_range = {"min": -30.0, "max": 30.0}

        if self.type == ActionSpaceType.DISCRETE and self.discrete_actions is None:
            self.discrete_actions = {"speed_levels": [1.0, 2.0, 3.0], "steering_angles": [-30.0, -15.0, 0.0, 15.0, 30.0]}

    def validate(self) -> bool:
        """Validate action space configuration.

        Returns
        -------
        bool
            True if configuration is valid
        """
        if self.type not in [ActionSpaceType.CONTINUOUS, ActionSpaceType.DISCRETE]:
            return False

        if self.type == ActionSpaceType.CONTINUOUS:
            if self.speed_range and (
                self.speed_range.get("min", 0) < 0 or self.speed_range.get("max", 0) <= self.speed_range.get("min", 0)
            ):
                return False

            if self.steering_range and (self.steering_range.get("max", 0) <= self.steering_range.get("min", 0)):
                return False

        if self.type == ActionSpaceType.DISCRETE:
            if self.num_speed_levels < 1 or self.num_steering_levels < 1:
                return False

        return True

    def generate_action_space(self) -> List[Dict[str, float]]:
        """Generate action space for AWS DeepRacer.

        Returns
        -------
        List[Dict[str, float]]
            List of action definitions
        """
        if self.type == ActionSpaceType.CONTINUOUS:
            return [
                {
                    "speed": {"min": self.speed_range.get("min", 0.5), "max": self.speed_range.get("max", 4.0)},
                    "steering_angle": {
                        "min": self.steering_range.get("min", -30.0),
                        "max": self.steering_range.get("max", 30.0),
                    },
                }
            ]

        actions = []

        if self.discrete_actions:
            speed_levels = self.discrete_actions.get("speed_levels", [1.0, 2.0, 3.0])
            steering_angles = self.discrete_actions.get("steering_angles", [-30.0, -15.0, 0.0, 15.0, 30.0])
        else:
            speed_min = self.speed_range.get("min", 1.0) if self.speed_range else 1.0
            speed_max = self.speed_range.get("max", 4.0) if self.speed_range else 4.0
            steering_min = self.steering_range.get("min", -30.0) if self.steering_range else -30.0
            steering_max = self.steering_range.get("max", 30.0) if self.steering_range else 30.0

            speed_levels = [
                speed_min + (speed_max - speed_min) * i / (self.num_speed_levels - 1) for i in range(self.num_speed_levels)
            ]
            steering_angles = [
                steering_min + (steering_max - steering_min) * i / (self.num_steering_levels - 1)
                for i in range(self.num_steering_levels)
            ]

        for speed in speed_levels:
            for steering in steering_angles:
                actions.append({"speed": round(speed, 2), "steering_angle": round(steering, 1)})

        return actions

    def get_action_space_size(self) -> int:
        """Get the size of the action space.

        Returns
        -------
        int
            Number of possible actions
        """
        if self.type == ActionSpaceType.CONTINUOUS:
            return -1

        return self.num_speed_levels * self.num_steering_levels

    def get_speed_range_tuple(self) -> Tuple[float, float]:
        """Get speed range as tuple for compatibility.

        Returns
        -------
        Tuple[float, float]
            (min_speed, max_speed)
        """
        if self.speed_range:
            return (self.speed_range["min"], self.speed_range["max"])
        return (0.5, 4.0)

    def get_steering_range_tuple(self) -> Tuple[float, float]:
        """Get steering range as tuple for compatibility.

        Returns
        -------
        Tuple[float, float]
            (min_steering, max_steering)
        """
        if self.steering_range:
            return (self.steering_range["min"], self.steering_range["max"])
        return (-30.0, 30.0)

    @property
    def continuous_actions(self) -> bool:
        """Check if action space is continuous.

        Returns
        -------
        bool
            True if action space is continuous
        """
        return self.type == ActionSpaceType.CONTINUOUS

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Action space configuration as dictionary
        """
        result = {"type": self.type.value if hasattr(self.type, "value") else self.type}

        if self.speed_range:
            result["speed_range"] = self.speed_range
        if self.steering_range:
            result["steering_range"] = self.steering_range
        if self.discrete_actions:
            result["discrete_actions"] = self.discrete_actions

        return result


DEFAULT_ACTION_SPACE = ActionSpaceConfig()

ActionSpaceConfiguration = ActionSpaceConfig
