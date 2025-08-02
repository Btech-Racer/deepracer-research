from enum import Enum, unique


@unique
class ObservationType(str, Enum):
    """Types of observations for DeepRacer training"""

    CAMERA = "CAMERA"
    LIDAR = "LIDAR"
    FRONT_FACING_CAMERA = "FRONT_FACING_CAMERA"
    LEFT_CAMERA = "LEFT_CAMERA"
    RIGHT_CAMERA = "RIGHT_CAMERA"
    CAMERA_LIDAR = "CAMERA_LIDAR"
    STEREO_CAMERA = "STEREO_CAMERA"
    FEATURES = "FEATURES"

    @classmethod
    def get_recommended_for_scenario(cls, scenario: str) -> "ObservationType":
        """Get recommended observation type for a training scenario.

        Parameters
        ----------
        scenario : str
            Training scenario name

        Returns
        -------
        ObservationType
            Recommended observation type
        """
        recommendations = {
            "centerline_following": cls.CAMERA,
            "object_avoidance": cls.CAMERA_LIDAR,
            "speed_optimization": cls.FRONT_FACING_CAMERA,
            "precision_driving": cls.STEREO_CAMERA,
            "simple_control": cls.FEATURES,
            "research": cls.CAMERA_LIDAR,
            "lidar_only": cls.LIDAR,
        }
        return recommendations.get(scenario, cls.CAMERA)

    def get_input_shape(self) -> tuple:
        """Get typical input shape for this observation type.

        Returns
        -------
        tuple
            Expected input shape (height, width, channels) or (features,)
        """
        shapes = {
            self.CAMERA: (160, 120, 3),
            self.LIDAR: (1080,),
            self.FRONT_FACING_CAMERA: (160, 120, 3),
            self.LEFT_CAMERA: (160, 120, 3),
            self.RIGHT_CAMERA: (160, 120, 3),
            self.CAMERA_LIDAR: (160, 120, 4),
            self.STEREO_CAMERA: (160, 120, 6),
            self.FEATURES: (10,),
        }
        return shapes[self]

    def requires_preprocessing(self) -> bool:
        """Check if this observation type requires preprocessing.

        Returns
        -------
        bool
            True if preprocessing is typically needed
        """
        preprocessing_required = {
            self.CAMERA: True,
            self.LIDAR: True,
            self.FRONT_FACING_CAMERA: True,
            self.LEFT_CAMERA: True,
            self.RIGHT_CAMERA: True,
            self.CAMERA_LIDAR: True,
            self.STEREO_CAMERA: True,
            self.FEATURES: False,
        }
        return preprocessing_required[self]

    def get_description(self) -> str:
        """Get a  description of the observation type.

        Returns
        -------
        str
            Description of the observation type
        """
        descriptions = {
            self.CAMERA: "RGB camera images (standard visual input)",
            self.LIDAR: "LiDAR distance measurements (precise ranging)",
            self.FRONT_FACING_CAMERA: "Front-facing camera (forward vision)",
            self.LEFT_CAMERA: "Left-side camera (left perspective)",
            self.RIGHT_CAMERA: "Right-side camera (right perspective)",
            self.CAMERA_LIDAR: "Combined camera and LiDAR (multimodal)",
            self.STEREO_CAMERA: "Stereo camera for depth (3D perception)",
            self.FEATURES: "Engineered features (simplified input)",
        }
        return descriptions[self]

    def get_computational_complexity(self) -> str:
        """Get computational complexity level for this observation type.

        Returns
        -------
        str
            Complexity level: 'low', 'medium', 'high', 'very_high'
        """
        complexity = {
            self.FEATURES: "low",
            self.CAMERA: "medium",
            self.LIDAR: "medium",
            self.FRONT_FACING_CAMERA: "medium",
            self.LEFT_CAMERA: "medium",
            self.RIGHT_CAMERA: "medium",
            self.CAMERA_LIDAR: "high",
            self.STEREO_CAMERA: "very_high",
        }
        return complexity[self]
