from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class ModelVersion:
    """Model version tracking for systematic management

    Parameters
    ----------
    version_id : str
        Version identifier
    parent_version : str, optional
        Parent version identifier, by default ""
    changes : str, optional
        Description of changes made, by default ""
    performance_delta : Dict[str, float], optional
        Performance changes from parent, by default empty dict
    is_active : bool, optional
        Whether this version is active, by default True
    created_date : datetime, optional
        Version creation timestamp, by default current datetime
    """

    version_id: str
    parent_version: str = ""
    changes: str = ""
    performance_delta: Dict[str, float] = field(default_factory=dict)
    is_active: bool = True
    created_date: datetime = field(default_factory=datetime.now)

    def deactivate(self) -> None:
        """Deactivate this model version.

        Returns
        -------
        None
        """
        self.is_active = False

    def activate(self) -> None:
        """Activate this model version.

        Returns
        -------
        None
        """
        self.is_active = True

    def add_performance_delta(self, metric: str, delta: float) -> None:
        """Add a performance delta for a specific metric.

        Parameters
        ----------
        metric : str
            The performance metric name
        delta : float
            The change in performance (positive for improvement)

        Returns
        -------
        None
        """
        self.performance_delta[metric] = delta

    def get_performance_improvement(self, metric: str) -> Optional[float]:
        """Get the performance improvement for a specific metric.

        Parameters
        ----------
        metric : str
            The performance metric name

        Returns
        -------
        Optional[float]
            The performance delta for the metric, or None if not found
        """
        return self.performance_delta.get(metric)

    def has_parent(self) -> bool:
        """Check if this version has a parent version.

        Returns
        -------
        bool
            True if this version has a parent, False otherwise
        """
        return bool(self.parent_version.strip())

    def get_version_info(self) -> Dict[str, Any]:
        """Get comprehensive version information.

        Returns
        -------
        Dict[str, any]
            Dictionary containing version_id, parent_version, changes,
            performance_delta, is_active, and created_date
        """
        return {
            "version_id": self.version_id,
            "parent_version": self.parent_version,
            "changes": self.changes,
            "performance_delta": self.performance_delta,
            "is_active": self.is_active,
            "created_date": self.created_date,
        }
