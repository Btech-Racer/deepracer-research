from enum import StrEnum
from typing import Any, Dict, Optional


class DeploymentMode(StrEnum):
    """Deployment modes for NVIDIA Brev instances"""

    ON_DEMAND = "on-demand"

    SPOT = "spot"

    RESERVED = "reserved"

    PREEMPTIBLE = "preemptible"

    @property
    def display_name(self) -> str:
        """Get  display name.

        Returns
        -------
        str
            Display name for the deployment mode
        """
        _display_names = {
            self.ON_DEMAND: "On-Demand",
            self.SPOT: "Spot Instance",
            self.RESERVED: "Reserved Instance",
            self.PREEMPTIBLE: "Preemptible Instance",
        }
        return _display_names[self]

    @property
    def description(self) -> str:
        """Get detailed description of the deployment mode.

        Returns
        -------
        str
            Detailed description
        """
        _descriptions = {
            self.ON_DEMAND: "Pay-as-you-go with immediate availability and no time limits",
            self.SPOT: "Lower cost instances that may be interrupted when demand increases",
            self.RESERVED: "Discounted pricing in exchange for longer-term commitment",
            self.PREEMPTIBLE: "Lowest cost instances with maximum runtime limitations",
        }
        return _descriptions[self]

    @property
    def cost_multiplier(self) -> float:
        """Get relative cost multiplier compared to on-demand pricing.

        Returns
        -------
        float
            Cost multiplier (1.0 = same as on-demand, 0.5 = 50% cheaper)
        """
        _cost_multipliers = {
            self.ON_DEMAND: 1.0,
            self.SPOT: 0.3,
            self.RESERVED: 0.6,
            self.PREEMPTIBLE: 0.2,
        }
        return _cost_multipliers[self]

    @property
    def availability_guarantee(self) -> str:
        """Get availability guarantee level.

        Returns
        -------
        str
            Availability guarantee description
        """
        _guarantees = {
            self.ON_DEMAND: "High - Available when requested",
            self.SPOT: "Medium - May be interrupted",
            self.RESERVED: "High - Guaranteed for reserved period",
            self.PREEMPTIBLE: "Low - Limited runtime with preemption",
        }
        return _guarantees[self]

    @property
    def is_interruptible(self) -> bool:
        """Check if instances can be interrupted/preempted.

        Returns
        -------
        bool
            True if instances can be interrupted
        """
        return self in {self.SPOT, self.PREEMPTIBLE}

    @property
    def suitable_for_production(self) -> bool:
        """Check if suitable for production workloads.

        Returns
        -------
        bool
            True if suitable for production
        """
        return self in {self.ON_DEMAND, self.RESERVED}

    @property
    def suitable_for_training(self) -> bool:
        """Check if suitable for ML training workloads.

        Returns
        -------
        bool
            True if suitable for training (can handle checkpointing)
        """
        return True

    @property
    def max_runtime_hours(self) -> Optional[int]:
        """Get maximum runtime in hours, if applicable.

        Returns
        -------
        Optional[int]
            Maximum runtime in hours, None if unlimited
        """
        _max_runtimes = {
            self.ON_DEMAND: None,
            self.SPOT: None,
            self.RESERVED: None,
            self.PREEMPTIBLE: 24,
        }
        return _max_runtimes[self]

    def get_cost_estimate(self, base_hourly_cost: float, hours: int) -> float:
        """Calculate estimated cost for given duration.

        Parameters
        ----------
        base_hourly_cost : float
            Base hourly cost for on-demand pricing
        hours : int
            Number of hours to run

        Returns
        -------
        float
            Estimated total cost
        """
        return base_hourly_cost * self.cost_multiplier * hours

    @classmethod
    def get_recommended_for_deepracer(cls) -> "DeploymentMode":
        """Get recommended deployment mode for DeepRacer training.

        Returns
        -------
        DeploymentMode
            Recommended mode
        """
        return cls.SPOT

    @classmethod
    def get_default(cls) -> "DeploymentMode":
        """Get default deployment mode.

        Returns
        -------
        DeploymentMode
            Default mode (ON_DEMAND)
        """
        return cls.ON_DEMAND

    def to_dict(self) -> Dict[str, Any]:
        """Convert deployment mode information to dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation
        """
        return {
            "mode": self.value,
            "display_name": self.display_name,
            "description": self.description,
            "cost_multiplier": self.cost_multiplier,
            "availability_guarantee": self.availability_guarantee,
            "is_interruptible": self.is_interruptible,
            "suitable_for_production": self.suitable_for_production,
            "suitable_for_training": self.suitable_for_training,
            "max_runtime_hours": self.max_runtime_hours,
        }
