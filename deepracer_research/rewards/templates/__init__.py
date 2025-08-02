AVAILABLE_TEMPLATES = [
    "centerline_following",
    "speed_optimization",
    "object_avoidance",
    "object_avoidance_static",
    "object_avoidance_dynamic",
    "time_trial",
    "head_to_head",
]

TEMPLATE_NAMES = [
    "centerline_following",
    "speed_optimization",
    "object_avoidance",
    "object_avoidance_static",
    "object_avoidance_dynamic",
    "time_trial",
    "head_to_head",
]

TEMPLATE_SCENARIO_MAPPING = {
    "centerline_following": "CENTERLINE_FOLLOWING",
    "speed_optimization": "SPEED_OPTIMIZATION",
    "object_avoidance": "OBJECT_AVOIDANCE",
    "object_avoidance_static": "OBJECT_AVOIDANCE_STATIC",
    "object_avoidance_dynamic": "OBJECT_AVOIDANCE_DYNAMIC",
    "time_trial": "TIME_TRIAL",
    "head_to_head": "HEAD_TO_HEAD",
}

__all__ = ["AVAILABLE_TEMPLATES", "TEMPLATE_NAMES", "TEMPLATE_SCENARIO_MAPPING"]
