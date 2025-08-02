from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from deepracer_research.training.enums.training_job_status import TrainingJobStatus


@dataclass
class TrainingJobResult:
    """Results and metadata from a training job."""

    job_id: str
    model_name: str
    status: TrainingJobStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    model_arn: Optional[str] = None
    failure_reason: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    logs_location: Optional[str] = None
    artifacts_location: Optional[str] = None

    def update_from_aws_response(self, response: Dict[str, Any]):
        """Update result from AWS API response."""
        if "Status" in response:
            try:
                self.status = TrainingJobStatus(response["Status"])
            except ValueError:
                self.status = TrainingJobStatus.UNKNOWN

        if "ModelArn" in response:
            self.model_arn = response["ModelArn"]

        if "FailureReason" in response:
            self.failure_reason = response["FailureReason"]

        if "CreationTime" in response:
            self.start_time = response["CreationTime"]

        if "TrainingEndTime" in response:
            self.end_time = response["TrainingEndTime"]

        if self.start_time and self.end_time:
            duration = self.end_time - self.start_time
            self.duration_minutes = duration.total_seconds() / 60
