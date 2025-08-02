import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from deepracer_research.config.aws.aws_hyperparameters import AWSHyperparameters
from deepracer_research.experiments.evaluation.metrics.performance_metrics import PerformanceMetrics


@dataclass
class EvaluationResults:
    """Results from model evaluation sessions."""

    evaluation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    model_name: str = ""
    track_name: str = ""
    evaluation_date: datetime = field(default_factory=datetime.now)

    num_evaluation_episodes: int = 20
    max_episode_time: int = 180
    hyperparameters: Optional[AWSHyperparameters] = None

    metrics: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    episode_data: List[Dict[str, Any]] = field(default_factory=list)
    lap_times: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)

    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    statistical_significance: Dict[str, float] = field(default_factory=dict)

    def add_episode_result(self, episode_data: Dict[str, Any]):
        """Add results from a single evaluation episode."""
        self.episode_data.append(episode_data)

        if "reward" in episode_data:
            self.reward_history.append(episode_data["reward"])
        if "lap_time" in episode_data:
            self.lap_times.append(episode_data["lap_time"])

    def finalize_evaluation(self):
        """Finalize evaluation by calculating comprehensive metrics."""
        if not self.episode_data:
            return

        self.metrics.total_episodes = len(self.episode_data)

        if self.reward_history:
            self.metrics.total_reward = sum(self.reward_history)
            self.metrics.avg_reward_per_episode = np.mean(self.reward_history)
            self.metrics.max_reward = max(self.reward_history)
            self.metrics.min_reward = min(self.reward_history)
            self.metrics.reward_std = np.std(self.reward_history)

        if self.lap_times:
            self.metrics.best_lap_time = min(self.lap_times)
            self.metrics.avg_lap_time = np.mean(self.lap_times)

        completed_episodes = sum(1 for ep in self.episode_data if ep.get("completed", False))
        crashed_episodes = sum(1 for ep in self.episode_data if ep.get("crashed", False))

        self.metrics.successful_laps = completed_episodes
        self.metrics.crash_count = crashed_episodes
        self.metrics.completion_rate = completed_episodes / len(self.episode_data)

        self.metrics.calculate_derived_metrics()

        self._calculate_confidence_intervals()

    def _calculate_confidence_intervals(self, confidence_level: float = 0.95):
        """Calculate confidence intervals for key metrics."""
        1 - confidence_level

        if self.reward_history:
            reward_mean = np.mean(self.reward_history)
            reward_sem = stats.sem(self.reward_history)
            reward_ci = stats.t.interval(confidence_level, len(self.reward_history) - 1, loc=reward_mean, scale=reward_sem)
            self.confidence_intervals["reward"] = reward_ci

        if self.lap_times:
            lap_mean = np.mean(self.lap_times)
            lap_sem = stats.sem(self.lap_times)
            lap_ci = stats.t.interval(confidence_level, len(self.lap_times) - 1, loc=lap_mean, scale=lap_sem)
            self.confidence_intervals["lap_time"] = lap_ci
