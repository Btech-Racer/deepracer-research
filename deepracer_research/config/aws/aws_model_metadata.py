from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class AWSModelMetadata:
    """Metadata for AWS DeepRacer models."""

    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    framework: str = "deepracer-research"
    converted_by: str = "AWSModelBuilder"
    author: Optional[str] = None
    experiment_id: Optional[str] = None
    research_phase: Optional[str] = None
    dataset_version: Optional[str] = None
    training_duration: Optional[str] = None
    model_size_mb: Optional[float] = None
    performance_metrics: Optional[Dict[str, float]] = None
    tags: Optional[Dict[str, str]] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format.

        Returns
        -------
        Dict[str, Any]
            Metadata as dictionary
        """
        result = {"created_at": self.created_at, "framework": self.framework, "converted_by": self.converted_by}

        optional_fields = [
            "author",
            "experiment_id",
            "research_phase",
            "dataset_version",
            "training_duration",
            "model_size_mb",
            "performance_metrics",
            "tags",
            "notes",
        ]

        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

        return result


DEFAULT_METADATA = AWSModelMetadata()
