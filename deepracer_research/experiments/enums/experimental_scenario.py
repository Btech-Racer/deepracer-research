from enum import Enum, unique


@unique
class ExperimentalScenario(str, Enum):
    """Enumeration of experimental scenarios for systematic evaluation"""

    CENTERLINE_FOLLOWING = "centerline_following"
    OBJECT_AVOIDANCE_STATIC = "object_avoidance_static"
    OBJECT_AVOIDANCE_DYNAMIC = "object_avoidance_dynamic"
    OBJECT_AVOIDANCE = "object_avoidance"
    SPEED_OPTIMIZATION = "speed_optimization"
    TIME_TRIAL = "time_trial"
    HEAD_TO_HEAD = "head_to_head"
    BASIC_FALLBACK = "basic_fallback"

    def to_deepracer_race_type(self) -> str:
        """Map experimental scenario to AWS DeepRacer race type.

        Returns
        -------
        str
            AWS DeepRacer race type string
        """
        scenario_to_race_type = {
            self.CENTERLINE_FOLLOWING: "TIME_TRIAL",
            self.OBJECT_AVOIDANCE_STATIC: "OBJECT_AVOIDANCE",
            self.OBJECT_AVOIDANCE_DYNAMIC: "OBJECT_AVOIDANCE",
            self.OBJECT_AVOIDANCE: "OBJECT_AVOIDANCE",
            self.SPEED_OPTIMIZATION: "TIME_TRIAL",
            self.TIME_TRIAL: "TIME_TRIAL",
            self.HEAD_TO_HEAD: "HEAD_TO_BOT",
            self.BASIC_FALLBACK: "TIME_TRIAL",
        }
        return scenario_to_race_type.get(self, "TIME_TRIAL")
