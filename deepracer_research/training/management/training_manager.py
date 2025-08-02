import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError, NoCredentialsError

from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion
from deepracer_research.training.config.training_job_config import TrainingJobConfig
from deepracer_research.training.config.training_job_result import TrainingJobResult
from deepracer_research.training.enums.training_job_status import TrainingJobStatus
from deepracer_research.training.monitoring.training_job_monitor import TrainingJobMonitor
from deepracer_research.utils import error, get_deepracer_client, info, warning


class TrainingManager:
    """Comprehensive training management system for AWS DeepRacer research."""

    def __init__(self, aws_session=None, region_name: str = AWSRegion.US_EAST_1, storage_path: Optional[Path] = None):
        """Initialize the training manager."""
        self.region_name = region_name
        self.storage_path = storage_path or Path.cwd() / "training_jobs"
        self.storage_path.mkdir(exist_ok=True)

        try:
            self.deepracer_client = get_deepracer_client(region_name=region_name, aws_session=aws_session)
        except (NoCredentialsError, Exception) as e:
            warning(f"AWS client initialization failed: {e}")
            self.deepracer_client = None

        self.submitted_jobs: Dict[str, TrainingJobConfig] = {}
        self.job_results: Dict[str, TrainingJobResult] = {}
        self.monitor = TrainingJobMonitor(aws_session, region_name)

        self.monitor.register_callback("on_completion", self._on_job_completion)
        self.monitor.register_callback("on_failure", self._on_job_failure)

    def create_training_job(self, config: TrainingJobConfig) -> str:
        """Create and submit a training job to AWS DeepRacer."""
        if not self.deepracer_client:
            raise RuntimeError("AWS DeepRacer client not available")

        job_id = str(uuid.uuid4())[:8]

        try:
            request = config.to_aws_request()
            self.deepracer_client.create_training_job(**request)

            self.submitted_jobs[job_id] = config
            self.monitor.add_job(job_id, config.model_name)

            self._save_job_config(job_id, config)

            info(f"Created training job {job_id} for model {config.model_name}")
            return job_id

        except ClientError as e:
            error(f"Failed to create training job: {e}")
            raise

    def create_batch_jobs(self, configs: List[TrainingJobConfig], max_concurrent: int = 3) -> List[str]:
        """Create multiple training jobs with concurrency control."""
        job_ids = []

        def submit_job(config):
            try:
                job_id = self.create_training_job(config)
                return job_id
            except Exception as e:
                error(f"Failed to submit job for {config.model_name}: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [executor.submit(submit_job, config) for config in configs]

            for future in futures:
                result = future.result()
                if result:
                    job_ids.append(result)

        info(f"Submitted {len(job_ids)} training jobs successfully")
        return job_ids

    def stop_training_job(self, job_id: str) -> bool:
        """Stop a running training job."""
        if job_id not in self.submitted_jobs:
            error(f"Job {job_id} not found")
            return False

        config = self.submitted_jobs[job_id]

        try:
            self.deepracer_client.stop_training_job(ModelName=config.model_name)
            info(f"Stopped training job {job_id}")
            return True
        except ClientError as e:
            error(f"Failed to stop job {job_id}: {e}")
            return False

    def start_monitoring(self, interval_seconds: int = 30):
        """Start monitoring all training jobs."""
        self.monitor.start_monitoring(interval_seconds)

    def stop_monitoring(self):
        """Stop monitoring training jobs."""
        self.monitor.stop_monitoring()

    def get_job_status(self, job_id: str) -> Optional[TrainingJobResult]:
        """Get current status of a training job."""
        return self.monitor.get_job_status(job_id)

    def get_all_job_statuses(self) -> Dict[str, TrainingJobResult]:
        """Get status of all managed jobs."""
        return self.monitor.get_all_jobs()

    def list_completed_jobs(self) -> List[TrainingJobResult]:
        """Get list of completed training jobs."""
        return [result for result in self.job_results.values() if result.status == TrainingJobStatus.COMPLETED]

    def list_failed_jobs(self) -> List[TrainingJobResult]:
        """Get list of failed training jobs."""
        return [result for result in self.job_results.values() if result.status == TrainingJobStatus.FAILED]

    def export_job_summary(self, filepath: str):
        """Export summary of all jobs to JSON file."""
        summary = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_jobs": len(self.submitted_jobs),
                "completed_jobs": len(self.list_completed_jobs()),
                "failed_jobs": len(self.list_failed_jobs()),
            },
            "jobs": [],
        }

        for job_id, config in self.submitted_jobs.items():
            result = self.job_results.get(job_id)
            job_data = {
                "job_id": job_id,
                "model_name": config.model_name,
                "racing_track": config.racing_track,
                "neural_network": config.neural_network,
                "sensors": config.sensors,
                "status": result.status.value if result else "UNKNOWN",
                "duration_minutes": result.duration_minutes if result else None,
            }
            summary["jobs"].append(job_data)

        with open(filepath, "w") as f:
            json.dump(summary, f, indent=2)

        info(f"Exported job summary to {filepath}")

    def _save_job_config(self, job_id: str, config: TrainingJobConfig):
        """Save job configuration to local storage."""
        config_path = self.storage_path / f"{job_id}_config.json"

        config_data = {
            "job_id": job_id,
            "model_name": config.model_name,
            "created_at": datetime.now().isoformat(),
            "config": {
                "racing_track": config.racing_track,
                "training_algorithm": config.training_algorithm,
                "neural_network": config.neural_network,
                "sensors": config.sensors,
                "hyperparameters": config.hyperparameters,
                "training_time_minutes": config.training_time_minutes,
                "action_space_size": len(config.action_space),
            },
        }

        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    def _on_job_completion(self, result: TrainingJobResult):
        """Handle job completion."""
        self.job_results[result.job_id] = result
        info(f"Job {result.job_id} completed successfully")

    def _on_job_failure(self, result: TrainingJobResult):
        """Handle job failure."""
        self.job_results[result.job_id] = result
        error(f"Job {result.job_id} failed: {result.failure_reason}")

    def get_training_capabilities(self) -> Dict[str, Any]:
        """Get current training system capabilities."""
        return {
            "aws_available": self.deepracer_client is not None,
            "client_connected": self.deepracer_client is not None,
            "region": self.region_name,
            "monitoring_active": self.monitor.monitoring_active,
            "total_jobs_submitted": len(self.submitted_jobs),
            "completed_jobs": len(self.list_completed_jobs()),
            "failed_jobs": len(self.list_failed_jobs()),
            "storage_path": str(self.storage_path),
        }
