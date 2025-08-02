from dataclasses import dataclass
from typing import Dict


@dataclass
class SensorConfig:
    """Configuration for AWS DeepRacer sensors."""

    CAMERA: bool = True
    LIDAR: bool = False
    FRONT_FACING_CAMERA: bool = True
    LEFT_CAMERA: bool = False
    RIGHT_CAMERA: bool = False

    def validate(self) -> bool:
        """Validate sensor configuration.

        Returns
        -------
        bool
            True if configuration is valid
        """
        cameras = [self.CAMERA, self.FRONT_FACING_CAMERA, self.LEFT_CAMERA, self.RIGHT_CAMERA]
        if not any(cameras):
            return False
        return True

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary format.

        Returns
        -------
        Dict[str, bool]
            Sensor configuration as dictionary
        """
        return {
            "CAMERA": self.CAMERA,
            "LIDAR": self.LIDAR,
            "FRONT_FACING_CAMERA": self.FRONT_FACING_CAMERA,
            "LEFT_CAMERA": self.LEFT_CAMERA,
            "RIGHT_CAMERA": self.RIGHT_CAMERA,
        }


DEFAULT_SENSOR_CONFIG = SensorConfig()
