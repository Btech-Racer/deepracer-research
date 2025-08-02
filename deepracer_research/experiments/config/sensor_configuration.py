from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from deepracer_research.experiments.enums.sensor_modality import SensorModality


@dataclass
class SensorConfiguration:
    """Configuration for sensor modalities and parameters."""

    modality: SensorModality
    resolution: Tuple[int, int] = (160, 120)
    field_of_view: float = 62.2
    range_max: float = 5.0
    update_frequency: int = 15
    noise_model: Optional[str] = None
    calibration_params: Optional[Dict[str, float]] = None

    def to_aws_format(self) -> List[str]:
        """Convert sensor configuration to AWS DeepRacer format."""
        aws_sensors = {
            SensorModality.MONOCULAR_CAMERA: ["FRONT_FACING_CAMERA"],
            SensorModality.STEREO_CAMERA: ["FRONT_FACING_CAMERA", "LEFT_CAMERA", "RIGHT_CAMERA"],
            SensorModality.LIDAR_FUSION: ["FRONT_FACING_CAMERA", "LIDAR"],
            SensorModality.RGB_CAMERA: ["FRONT_FACING_CAMERA"],
            SensorModality.FRONT_FACING_CAMERA: ["FRONT_FACING_CAMERA"],
        }
        return aws_sensors.get(self.modality, ["FRONT_FACING_CAMERA"])
