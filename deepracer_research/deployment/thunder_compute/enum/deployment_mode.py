from enum import StrEnum
from typing import Dict


class DeploymentMode(StrEnum):
    """Available deployment modes for Thunder Compute instances"""

    PROTOTYPING = "prototyping"

    PRODUCTION = "production"

    @property
    def description(self) -> str:
        """Get the description of the deployment mode.

        Returns
        -------
        str
            Description of the deployment mode.
        """
        _descriptions = {
            self.PROTOTYPING: "Development and prototyping environment with lower cost",
            self.PRODUCTION: "Stable production environment with guaranteed availability",
        }
        return _descriptions[self]

    @property
    def is_cost_optimized(self) -> bool:
        """Check if deployment mode is optimized for cost.

        Returns
        -------
        bool
            True if mode prioritizes cost savings over availability.
        """
        return self == self.PROTOTYPING

    @property
    def has_guaranteed_availability(self) -> bool:
        """Check if deployment mode guarantees instance availability.

        Returns
        -------
        bool
            True if mode provides guaranteed availability SLA.
        """
        return self == self.PRODUCTION

    @classmethod
    def get_recommended_for_research(cls) -> "DeploymentMode":
        """Get the recommended deployment mode for research workloads.

        Returns
        -------
        DeploymentMode
            Recommended mode for research and development work.
        """
        return cls.PROTOTYPING

    @classmethod
    def get_recommended_for_training(cls) -> "DeploymentMode":
        """Get the recommended deployment mode for training workloads.

        Returns
        -------
        DeploymentMode
            Recommended mode for long-running training jobs.
        """
        return cls.PRODUCTION

    @classmethod
    def get_all_info(cls) -> Dict[str, Dict[str, any]]:
        """Get comprehensive information about all deployment modes.

        Returns
        -------
        Dict[str, Dict[str, any]]
            Dictionary mapping modes to their characteristics.
        """
        return {
            mode.value: {
                "description": mode.description,
                "cost_optimized": mode.is_cost_optimized,
                "guaranteed_availability": mode.has_guaranteed_availability,
            }
            for mode in cls
        }
