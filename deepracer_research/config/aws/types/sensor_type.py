from enum import Enum, unique
from typing import List


@unique
class SensorType(str, Enum):
    """Sensor types supported by AWS DeepRacer.

    Based on AWS DeepRacer community wisdom and performance analysis.
    Reference: https://blog.gofynd.com/how-we-broke-into-the-top-1-of-the-aws-deepracer-virtual-circuit-573ba46c275
    """

    FRONT_FACING_CAMERA = "FRONT_FACING_CAMERA"
    """Front-facing camera sensor (Recommended for Time Trials).

    Primary sensor for most racing scenarios. Provides RGB/grayscale image input.
    Proven effective for top 1% performance in time trial competitions.
    Fastest training and inference times.
    """

    CAMERA = "CAMERA"
    """Generic camera sensor (legacy compatibility)."""

    LEFT_CAMERA = "LEFT_CAMERA"
    """Left-side camera for stereo vision setups."""

    RIGHT_CAMERA = "RIGHT_CAMERA"
    """Right-side camera for stereo vision setups."""

    STEREO_CAMERAS = "STEREO_CAMERAS"
    """Stereo camera setup for depth perception.

    Two cameras providing depth information for advanced scenarios.
    Higher computational cost but better spatial awareness.
    """

    LIDAR = "LIDAR"
    """Light Detection and Ranging sensor.

    Provides precise distance measurements in 360 degrees.
    Excellent for object avoidance and complex environments.
    Higher computational and training cost.
    """

    SECTOR_LIDAR = "SECTOR_LIDAR"
    """Sector-based LIDAR sensor configuration.

    Provides distance measurements in specific sectors/zones around the vehicle.
    More efficient than full 360° LIDAR while maintaining spatial awareness.
    Good balance between computational cost and environmental perception.
    """

    FRONT_FACING_CAMERA_LIDAR = "FRONT_FACING_CAMERA_LIDAR"
    """Combination of front camera and LIDAR.

    Multi-modal sensing for complex scenarios requiring both
    visual and spatial information. Maximum computational cost.
    """

    @classmethod
    def get_recommended_for_time_trials(cls) -> "SensorType":
        """Get the recommended sensor for time trial competitions.

        Returns
        -------
        SensorType
            FRONT_FACING_CAMERA - proven optimal for time trials
        """
        return cls.FRONT_FACING_CAMERA

    @classmethod
    def get_for_object_avoidance(cls) -> "SensorType":
        """Get the optimal sensor for object avoidance scenarios.

        Returns
        -------
        SensorType
            SECTOR_LIDAR - efficient obstacle detection with good performance balance
        """
        return cls.SECTOR_LIDAR

    @classmethod
    def get_for_head_to_head(cls) -> "SensorType":
        """Get the optimal sensor for head-to-head racing.

        Returns
        -------
        SensorType
            FRONT_FACING_CAMERA_LIDAR - comprehensive sensing for competitive racing
        """
        return cls.FRONT_FACING_CAMERA_LIDAR

    @classmethod
    def get_minimal_setup(cls) -> "SensorType":
        """Get the minimal sensor setup for fastest training.

        Returns
        -------
        SensorType
            FRONT_FACING_CAMERA - minimal computational overhead
        """
        return cls.FRONT_FACING_CAMERA

    def get_sensor_list(self) -> List[str]:
        """Get the sensor configuration as a list for AWS metadata.

        Returns
        -------
        List[str]
            List of sensor types for AWS DeepRacer configuration
        """
        return [self.value]

    @classmethod
    def get_sensor_list_from_types(cls, sensor_types: List["SensorType"]) -> List[str]:
        """Get sensor configuration list from multiple SensorType objects.

        Parameters
        ----------
        sensor_types : List[SensorType]
            List of sensor type objects

        Returns
        -------
        List[str]
            List of sensor type strings for AWS DeepRacer configuration
        """
        return [sensor.value for sensor in sensor_types]

    def get_description(self) -> str:
        """Get  description of the sensor type.

        Returns
        -------
        str
            Description of the sensor configuration
        """
        descriptions = {
            self.FRONT_FACING_CAMERA: "Single front camera - fast training, recommended for time trials",
            self.CAMERA: "Generic camera sensor - legacy compatibility",
            self.LEFT_CAMERA: "Left camera - for stereo vision setups",
            self.RIGHT_CAMERA: "Right camera - for stereo vision setups",
            self.STEREO_CAMERAS: "Dual cameras - depth perception, moderate computational cost",
            self.LIDAR: "360° distance sensor - excellent object detection, higher cost",
            self.SECTOR_LIDAR: "Sector-based LIDAR - efficient spatial awareness, balanced cost",
            self.FRONT_FACING_CAMERA_LIDAR: "Camera + LIDAR - comprehensive sensing, maximum cost",
        }
        return descriptions[self]

    def get_training_time_multiplier(self) -> float:
        """Get approximate training time multiplier compared to front camera.

        Returns
        -------
        float
            Training time multiplier (1.0 = baseline front camera)
        """
        multipliers = {
            self.FRONT_FACING_CAMERA: 1.0,
            self.CAMERA: 1.0,
            self.LEFT_CAMERA: 1.2,
            self.RIGHT_CAMERA: 1.2,
            self.STEREO_CAMERAS: 1.5,
            self.LIDAR: 2.0,
            self.SECTOR_LIDAR: 1.7,
            self.FRONT_FACING_CAMERA_LIDAR: 3.0,
        }
        return multipliers[self]

    def is_recommended_for_competitions(self) -> bool:
        """Check if this sensor is recommended for competitions.

        Returns
        -------
        bool
            True if recommended for competitive racing
        """
        return self == self.FRONT_FACING_CAMERA

    @classmethod
    def validate_sensor_combination(cls, sensors: List[str]) -> tuple[bool, str]:
        """Validate that a combination of sensors is compatible.

        Parameters
        ----------
        sensors : List[str]
            List of sensor type strings to validate

        Returns
        -------
        tuple[bool, str]
            (is_valid, error_message) - True if valid, False with error message if invalid
        """
        if not sensors:
            return False, "At least one sensor must be specified"

        try:
            sensor_types = [cls(sensor) for sensor in sensors]
        except ValueError as e:
            return False, f"Invalid sensor type: {e}"

        incompatible_pairs = [
            (cls.STEREO_CAMERAS, cls.FRONT_FACING_CAMERA),
            (cls.STEREO_CAMERAS, cls.CAMERA),
            (cls.LIDAR, cls.SECTOR_LIDAR),
            (cls.FRONT_FACING_CAMERA, cls.CAMERA),
        ]

        for sensor1, sensor2 in incompatible_pairs:
            if sensor1 in sensor_types and sensor2 in sensor_types:
                return False, f"Incompatible sensor combination: {sensor1.value} and {sensor2.value} cannot be used together"

        if cls.FRONT_FACING_CAMERA_LIDAR in sensor_types:
            if any(s in sensor_types for s in [cls.FRONT_FACING_CAMERA, cls.LIDAR, cls.SECTOR_LIDAR, cls.CAMERA]):
                return (
                    False,
                    "FRONT_FACING_CAMERA_LIDAR is a complete sensor configuration and cannot be combined with individual camera or LIDAR sensors",
                )
        if len(sensor_types) > 2:
            return False, f"Too many sensors specified ({len(sensor_types)}). Maximum of 2 sensors allowed"

        return True, ""

    @classmethod
    def get_compatible_sensors(cls, base_sensor: "SensorType") -> List["SensorType"]:
        """Get list of sensors compatible with the given base sensor.

        Parameters
        ----------
        base_sensor : SensorType
            The base sensor to find compatible sensors for

        Returns
        -------
        List[SensorType]
            List of compatible sensor types
        """
        compatibility_map = {
            cls.FRONT_FACING_CAMERA: [cls.FRONT_FACING_CAMERA],
            cls.CAMERA: [cls.CAMERA],
            cls.LEFT_CAMERA: [cls.LEFT_CAMERA, cls.RIGHT_CAMERA],
            cls.RIGHT_CAMERA: [cls.LEFT_CAMERA, cls.RIGHT_CAMERA],
            cls.STEREO_CAMERAS: [cls.STEREO_CAMERAS],
            cls.LIDAR: [cls.LIDAR],
            cls.SECTOR_LIDAR: [cls.SECTOR_LIDAR, cls.FRONT_FACING_CAMERA],
            cls.FRONT_FACING_CAMERA_LIDAR: [cls.FRONT_FACING_CAMERA_LIDAR],
        }

        return compatibility_map.get(base_sensor, [base_sensor])

    @classmethod
    def parse_sensor_list(cls, sensor_string: str) -> List["SensorType"]:
        """Parse a comma-separated string of sensor types.

        Parameters
        ----------
        sensor_string : str
            Comma-separated sensor type names

        Returns
        -------
        List[SensorType]
            List of parsed sensor types

        Raises
        ------
        ValueError
            If any sensor type is invalid or combination is incompatible
        """
        if not sensor_string.strip():
            raise ValueError("Sensor string cannot be empty")

        sensor_names = [s.strip() for s in sensor_string.split(",") if s.strip()]

        unique_sensors = []
        for sensor in sensor_names:
            if sensor not in unique_sensors:
                unique_sensors.append(sensor)

        is_valid, error_msg = cls.validate_sensor_combination(unique_sensors)
        if not is_valid:
            raise ValueError(error_msg)

        return [cls(sensor) for sensor in unique_sensors]
