from enum import Enum


class EvaluationType(Enum):
    """Types of evaluation that can be performed in AWS DeepRacer."""

    TIME_TRIAL = "TIME_TRIAL"
    OBJECT_AVOIDANCE = "OBJECT_AVOIDANCE"
    HEAD_TO_HEAD = "HEAD_TO_HEAD"

    @classmethod
    def get_default(cls) -> "EvaluationType":
        """Get the default evaluation type."""
        return cls.TIME_TRIAL

    @property
    def description(self) -> str:
        """Get a  description of the evaluation type."""
        descriptions = {
            self.TIME_TRIAL: "Time trial evaluation on track without obstacles",
            self.OBJECT_AVOIDANCE: "Evaluation with static and dynamic obstacles",
            self.HEAD_TO_HEAD: "Head-to-head racing evaluation",
        }
        return descriptions.get(self, "Unknown evaluation type")

    @property
    def requires_obstacles(self) -> bool:
        """Check if this evaluation type requires obstacles."""
        return self in [self.OBJECT_AVOIDANCE]

    @property
    def supports_multiple_agents(self) -> bool:
        """Check if this evaluation type supports multiple agents."""
        return self in [self.HEAD_TO_HEAD]
