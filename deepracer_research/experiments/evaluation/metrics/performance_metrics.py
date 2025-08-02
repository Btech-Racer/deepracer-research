from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics for training and evaluation."""

    total_episodes: int = 0
    training_time_minutes: float = 0.0
    avg_episode_duration: float = 0.0
    completion_rate: float = 0.0
    best_lap_time: Optional[float] = None
    avg_lap_time: Optional[float] = None

    total_reward: float = 0.0
    avg_reward_per_episode: float = 0.0
    max_reward: float = 0.0
    min_reward: float = 0.0
    reward_std: float = 0.0

    track_completion_percentage: float = 0.0
    off_track_percentage: float = 0.0
    crash_count: int = 0
    successful_laps: int = 0

    avg_speed: float = 0.0
    max_speed: float = 0.0
    speed_variance: float = 0.0
    steering_smoothness: float = 0.0

    distance_traveled: float = 0.0
    energy_efficiency: float = 0.0
    path_efficiency: float = 0.0

    episodes_to_convergence: Optional[int] = None
    reward_convergence_value: Optional[float] = None
    training_stability: float = 0.0

    def calculate_derived_metrics(self):
        """Calculate derived metrics from base measurements."""
        if self.total_episodes > 0:
            self.avg_reward_per_episode = self.total_reward / self.total_episodes

        if self.distance_traveled > 0 and self.training_time_minutes > 0:
            self.energy_efficiency = self.distance_traveled / (self.training_time_minutes * 60)

        if self.successful_laps > 0 and self.total_episodes > 0:
            self.completion_rate = self.successful_laps / self.total_episodes

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {field.name: getattr(self, field.name) for field in self.__dataclass_fields__.values()}
