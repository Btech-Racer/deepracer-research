from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict


@dataclass
class ResearchConfig:
    """Main research configuration."""

    base_path: Path = field(default_factory=lambda: Path.cwd())
    data_path: Path = field(default_factory=lambda: Path.cwd() / "data")
    models_path: Path = field(default_factory=lambda: Path.cwd() / "models")
    logs_path: Path = field(default_factory=lambda: Path.cwd() / "logs")

    neural_architectures: Dict[str, Any] = field(
        default_factory=lambda: {
            "default_input_shape": (160, 120, 3),
            "default_num_actions": 2,
            "enable_gpu_optimization": True,
        }
    )

    logging: Dict[str, Any] = field(
        default_factory=lambda: {
            "level": "INFO",
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "file_handler": True,
            "console_handler": True,
        }
    )

    aws: Dict[str, Any] = field(
        default_factory=lambda: {"region": "us-east-1", "profile": "default", "sagemaker_role": None, "deepracer_bucket": None}
    )

    def __post_init__(self):
        """Ensure paths exist."""
        for path_attr in ["data_path", "models_path", "logs_path"]:
            path = getattr(self, path_attr)
            path.mkdir(parents=True, exist_ok=True)
