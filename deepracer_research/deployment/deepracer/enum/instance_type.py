from enum import Enum, unique


@unique
class DeepRacerInstanceType(str, Enum):
    """Supported instance types for DeepRacer training."""

    ML_C5_LARGE = "ml.c5.large"
    ML_C5_XLARGE = "ml.c5.xlarge"
    ML_C5_2XLARGE = "ml.c5.2xlarge"
    ML_C5_4XLARGE = "ml.c5.4xlarge"
    ML_M5_LARGE = "ml.m5.large"
    ML_M5_XLARGE = "ml.m5.xlarge"
    ML_M5_2XLARGE = "ml.m5.2xlarge"
