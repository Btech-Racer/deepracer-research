from enum import Enum, unique


@unique
class SensorModality(str, Enum):
    """Enumeration of sensor modalities for perception systems"""

    MONOCULAR_CAMERA = "monocular"

    STEREO_CAMERA = "stereo"

    LIDAR_FUSION = "lidar_fusion"

    RGB_CAMERA = "rgb_camera"

    FRONT_FACING_CAMERA = "front_facing_camera"
