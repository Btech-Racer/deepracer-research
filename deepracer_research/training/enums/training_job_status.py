from enum import Enum, unique


@unique
class TrainingJobStatus(str, Enum):
    """Enumeration of training job statuses."""

    PENDING = "PENDING"
    STARTING = "STARTING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    STOPPED = "STOPPED"
    UNKNOWN = "UNKNOWN"
