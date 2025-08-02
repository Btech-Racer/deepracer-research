from typing import Dict

from deepracer_research.rewards.parameters.parameter_definition import ParameterDefinition
from deepracer_research.rewards.parameters.parameter_type import ParameterType

CORE_PARAMETERS: Dict[str, ParameterDefinition] = {
    "all_wheels_on_track": ParameterDefinition(
        name="all_wheels_on_track",
        type=ParameterType.BOOLEAN,
        description="Flag to indicate if the agent is on the track",
        examples=["if not all_wheels_on_track: return 1e-3"],
    ),
    "x": ParameterDefinition(
        name="x",
        type=ParameterType.FLOAT,
        description="Agent's x-coordinate in meters",
        units="meters",
        examples=["distance = math.sqrt((x - waypoint_x)**2 + (y - waypoint_y)**2)"],
    ),
    "y": ParameterDefinition(
        name="y",
        type=ParameterType.FLOAT,
        description="Agent's y-coordinate in meters",
        units="meters",
        examples=["distance = math.sqrt((x - waypoint_x)**2 + (y - waypoint_y)**2)"],
    ),
    "distance_from_center": ParameterDefinition(
        name="distance_from_center",
        type=ParameterType.FLOAT,
        description="Distance in meters from the track center",
        units="meters",
        range_info="0 to track_width/2",
        examples=["reward = 1.0 - (distance_from_center / (track_width/2))"],
    ),
    "is_left_of_center": ParameterDefinition(
        name="is_left_of_center",
        type=ParameterType.BOOLEAN,
        description="Flag to indicate if the agent is on the left side of track center",
        examples=["if is_left_of_center: steering_bonus = 0.1"],
    ),
    "heading": ParameterDefinition(
        name="heading",
        type=ParameterType.FLOAT,
        description="Agent's yaw in degrees",
        units="degrees",
        range_info="-180 to 180",
        examples=["direction_diff = abs(heading - waypoint_heading)"],
    ),
    "progress": ParameterDefinition(
        name="progress",
        type=ParameterType.FLOAT,
        description="Percentage of track completed",
        units="percentage",
        range_info="0 to 100",
        examples=["if progress > 99: reward += 10.0"],
    ),
    "speed": ParameterDefinition(
        name="speed",
        type=ParameterType.FLOAT,
        description="Agent's speed in meters per second",
        units="m/s",
        range_info="0 to max_speed",
        examples=["speed_reward = speed / 4.0"],
    ),
    "steering_angle": ParameterDefinition(
        name="steering_angle",
        type=ParameterType.FLOAT,
        description="Agent's steering angle in degrees",
        units="degrees",
        range_info="-30 to 30",
        examples=["if abs(steering_angle) < 15: reward += 0.5"],
    ),
    "steps": ParameterDefinition(
        name="steps",
        type=ParameterType.INT,
        description="Number of steps completed",
        examples=["efficiency = progress / steps"],
    ),
    "track_length": ParameterDefinition(
        name="track_length",
        type=ParameterType.FLOAT,
        description="Track length in meters",
        units="meters",
        examples=["completion_rate = progress * track_length / 100"],
    ),
    "track_width": ParameterDefinition(
        name="track_width",
        type=ParameterType.FLOAT,
        description="Width of the track",
        units="meters",
        examples=["margin = track_width * 0.5"],
    ),
}
