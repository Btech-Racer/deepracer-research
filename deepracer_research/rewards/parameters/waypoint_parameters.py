from typing import Dict

from deepracer_research.rewards.parameters.parameter_definition import ParameterDefinition
from deepracer_research.rewards.parameters.parameter_type import ParameterType

WAYPOINT_PARAMETERS: Dict[str, ParameterDefinition] = {
    "closest_waypoints": ParameterDefinition(
        name="closest_waypoints",
        type=ParameterType.LIST_INT,
        description="Indices of the two nearest waypoints",
        examples=["next_point = waypoints[closest_waypoints[1]]"],
    ),
    "waypoints": ParameterDefinition(
        name="waypoints",
        type=ParameterType.LIST_TUPLE,
        description="List of (x,y) coordinates as milestones along the track center",
        examples=["next_waypoint = waypoints[closest_waypoints[1]]"],
    ),
}
