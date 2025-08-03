import threading
import time
from typing import Callable, Dict, List, Optional

from botocore.exceptions import NoCredentialsError

from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion
from deepracer_research.training.config.training_job_result import TrainingJobResult
from deepracer_research.training.enums.training_job_status import TrainingJobStatus
from deepracer_research.utils.aws_config import get_deepracer_client
from deepracer_research.utils.logger import error, info, warning


class TrainingJobMonitor:
    """Monitor training job progress and status."""

    def __init__(self, aws_session=None, region_name: str = AWSRegion.US_EAST_1):
        """Initialize the training job monitor."""
        self.region_name = region_name
        self.active_jobs: Dict[str, TrainingJobResult] = {}
        self.monitoring_active = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.callbacks: Dict[str, List[Callable]] = {"on_status_change": [], "on_completion": [], "on_failure": []}
        try:
            self.deepracer_client = get_deepracer_client(region_name=region_name, aws_session=aws_session)
        except (NoCredentialsError, Exception) as e:
            warning(f"AWS client initialization failed: {e}")
            self.deepracer_client = None

    def add_job(self, job_id: str, model_name: str):
        """Add a job to monitoring."""
        result = TrainingJobResult(job_id=job_id, model_name=model_name, status=TrainingJobStatus.PENDING)
        self.active_jobs[job_id] = result
        info(f"Added job {job_id} ({model_name}) to monitoring")

    def register_callback(self, event_type: str, callback: Callable):
        """Register callback for monitoring events."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)

    def start_monitoring(self, interval_seconds: int = 30):
        """Start monitoring active jobs."""
        if self.monitoring_active:
            warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval_seconds,), daemon=True)
        self.monitor_thread.start()
        info(f"Started monitoring with {interval_seconds}s interval")

    def stop_monitoring(self):
        """Stop monitoring jobs."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        info("Stopped monitoring")

    def _monitor_loop(self, interval_seconds: int):
        """Main monitoring loop."""
        while self.monitoring_active:
            if self.active_jobs and self.deepracer_client:
                self._update_job_statuses()
            time.sleep(interval_seconds)

    def _update_job_statuses(self):
        """Update status of all active jobs."""
        jobs_to_remove = []

        for job_id, result in self.active_jobs.items():
            try:
                response = self.deepracer_client.describe_training_job(ModelName=result.model_name)

                old_status = result.status
                result.update_from_aws_response(response)

                if old_status != result.status:
                    for callback in self.callbacks["on_status_change"]:
                        callback(result)

                    if result.status in [TrainingJobStatus.COMPLETED, TrainingJobStatus.FAILED, TrainingJobStatus.STOPPED]:
                        if result.status == TrainingJobStatus.COMPLETED:
                            for callback in self.callbacks["on_completion"]:
                                callback(result)
                        else:
                            for callback in self.callbacks["on_failure"]:
                                callback(result)

                        jobs_to_remove.append(job_id)

            except Exception as e:
                error(f"Error updating job {job_id}: {e}")

        for job_id in jobs_to_remove:
            del self.active_jobs[job_id]

    def get_job_status(self, job_id: str) -> Optional[TrainingJobResult]:
        """Get current status of a specific job."""
        return self.active_jobs.get(job_id)

    def get_all_jobs(self) -> Dict[str, TrainingJobResult]:
        """Get status of all monitored jobs."""
        return self.active_jobs.copy()
