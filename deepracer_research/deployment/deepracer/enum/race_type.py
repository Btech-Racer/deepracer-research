from enum import Enum, unique


@unique
class DeepRacerRaceType(str, Enum):
    """DeepRacer race types supported by the AWS API."""

    TIME_TRIAL = "TIME_TRIAL"
    OBJECT_AVOIDANCE = "OBJECT_AVOIDANCE"
    HEAD_TO_HEAD = "HEAD_TO_BOT"
