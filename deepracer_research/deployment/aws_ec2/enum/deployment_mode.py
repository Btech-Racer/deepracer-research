from enum import StrEnum
from typing import Any, Dict


class EC2DeploymentMode(StrEnum):
    """Deployment modes for EC2 instances affecting pricing and behavior"""

    ON_DEMAND = "on-demand"
    SPOT = "spot"
    RESERVED = "reserved"

    @property
    def display_name(self) -> str:
        """Get the display name for the deployment mode.

        Returns
        -------
        str
            Display name for the deployment mode.
        """
        _display_names = {
            self.ON_DEMAND: "On-Demand (Standard pricing, guaranteed availability)",
            self.SPOT: "Spot Instance (Up to 90% savings, may be interrupted)",
            self.RESERVED: "Reserved Instance (1-3 year commitment, significant savings)",
        }
        return _display_names[self]

    @property
    def cost_savings_percentage(self) -> int:
        """Get typical cost savings percentage compared to on-demand.

        Returns
        -------
        int
            Percentage savings (0-90).
        """
        _savings = {
            self.ON_DEMAND: 0,
            self.SPOT: 70,
            self.RESERVED: 60,
        }
        return _savings[self]

    @property
    def interruption_risk(self) -> str:
        """Get interruption risk level.

        Returns
        -------
        str
            Risk level description.
        """
        _risks = {
            self.ON_DEMAND: "None",
            self.SPOT: "High (can be interrupted with 2-minute notice)",
            self.RESERVED: "None",
        }
        return _risks[self]

    @property
    def suitable_for_training(self) -> bool:
        """Check if mode is suitable for training workloads.

        Returns
        -------
        bool
            True if mode is suitable for long-running training.
        """
        return self != self.SPOT

    @property
    def suitable_for_development(self) -> bool:
        """Check if mode is suitable for development workloads.

        Returns
        -------
        bool
            True if mode is suitable for development/testing.
        """
        return True

    @classmethod
    def get_recommended_for_training(cls) -> "EC2DeploymentMode":
        """Get the recommended mode for training workloads.

        Returns
        -------
        EC2DeploymentMode
            Recommended deployment mode for training.
        """
        return cls.ON_DEMAND

    @classmethod
    def get_recommended_for_development(cls) -> "EC2DeploymentMode":
        """Get the recommended mode for development workloads.

        Returns
        -------
        EC2DeploymentMode
            Recommended deployment mode for development.
        """
        return cls.SPOT

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, Any]]:
        """Get comprehensive information about all deployment modes.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary mapping modes to their specifications.
        """
        return {
            mode.value: {
                "display_name": mode.display_name,
                "cost_savings_percentage": mode.cost_savings_percentage,
                "interruption_risk": mode.interruption_risk,
                "suitable_for_training": mode.suitable_for_training,
                "suitable_for_development": mode.suitable_for_development,
            }
            for mode in cls
        }
