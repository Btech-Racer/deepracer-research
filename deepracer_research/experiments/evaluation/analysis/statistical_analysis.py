from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
from scipy import stats


@dataclass
class StatisticalAnalysis:
    """Statistical analysis results for experimental comparisons."""

    test_name: str = ""
    test_date: datetime = field(default_factory=datetime.now)
    significance_level: float = 0.05

    group1_name: str = ""
    group2_name: str = ""
    group1_data: List[float] = field(default_factory=list)
    group2_data: List[float] = field(default_factory=list)

    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    effect_size: Optional[float] = None
    power: Optional[float] = None

    group1_stats: Dict[str, float] = field(default_factory=dict)
    group2_stats: Dict[str, float] = field(default_factory=dict)

    reject_null: Optional[bool] = None
    conclusion: str = ""

    def perform_t_test(self, equal_var: bool = True):
        """Perform independent samples t-test."""
        if len(self.group1_data) < 2 or len(self.group2_data) < 2:
            raise ValueError("Insufficient data for t-test")

        self.test_name = "Independent Samples T-Test"

        self.group1_stats = {
            "mean": np.mean(self.group1_data),
            "std": np.std(self.group1_data, ddof=1),
            "n": len(self.group1_data),
        }

        self.group2_stats = {
            "mean": np.mean(self.group2_data),
            "std": np.std(self.group2_data, ddof=1),
            "n": len(self.group2_data),
        }

        self.test_statistic, self.p_value = stats.ttest_ind(self.group1_data, self.group2_data, equal_var=equal_var)

        pooled_std = np.sqrt(
            (
                (len(self.group1_data) - 1) * self.group1_stats["std"] ** 2
                + (len(self.group2_data) - 1) * self.group2_stats["std"] ** 2
            )
            / (len(self.group1_data) + len(self.group2_data) - 2)
        )

        self.effect_size = (self.group1_stats["mean"] - self.group2_stats["mean"]) / pooled_std

        self.reject_null = self.p_value < self.significance_level

        if self.reject_null:
            self.conclusion = f"Significant difference detected (p={self.p_value:.4f}, d={self.effect_size:.4f})"
        else:
            self.conclusion = f"No significant difference (p={self.p_value:.4f}, d={self.effect_size:.4f})"

    def perform_mann_whitney_u(self):
        """Perform Mann-Whitney U test (non-parametric alternative to t-test)."""
        if len(self.group1_data) < 2 or len(self.group2_data) < 2:
            raise ValueError("Insufficient data for Mann-Whitney U test")

        self.test_name = "Mann-Whitney U Test"

        self.group1_stats = {
            "median": np.median(self.group1_data),
            "mean": np.mean(self.group1_data),
            "n": len(self.group1_data),
        }

        self.group2_stats = {
            "median": np.median(self.group2_data),
            "mean": np.mean(self.group2_data),
            "n": len(self.group2_data),
        }

        self.test_statistic, self.p_value = stats.mannwhitneyu(self.group1_data, self.group2_data, alternative="two-sided")

        self.reject_null = self.p_value < self.significance_level

        if self.reject_null:
            self.conclusion = f"Significant difference detected (p={self.p_value:.4f})"
        else:
            self.conclusion = f"No significant difference (p={self.p_value:.4f})"
