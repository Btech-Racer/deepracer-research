from typing import Dict

from deepracer_research.rewards.parameters.parameter_definition import ParameterDefinition
from deepracer_research.rewards.parameters.parameter_type import ParameterType

OBJECT_PARAMETERS: Dict[str, ParameterDefinition] = {
    "closest_objects": ParameterDefinition(
        name="closest_objects",
        type=ParameterType.LIST_INT,
        description="Zero-based indices of the two closest objects to agent's position",
        examples=["if closest_objects[0] != -1: # Object detected"],
    ),
    "objects_distance": ParameterDefinition(
        name="objects_distance",
        type=ParameterType.LIST_FLOAT,
        description="List of objects' distances in meters from starting line",
        units="meters",
        range_info="0 to track_length",
        examples=["min_distance = min(objects_distance) if objects_distance else float('inf')"],
    ),
    "objects_heading": ParameterDefinition(
        name="objects_heading",
        type=ParameterType.LIST_FLOAT,
        description="List of objects' headings in degrees",
        units="degrees",
        range_info="-180 to 180",
        examples=["relative_heading = abs(heading - objects_heading[0])"],
    ),
    "objects_left_of_center": ParameterDefinition(
        name="objects_left_of_center",
        type=ParameterType.LIST_BOOLEAN,
        description="List of Boolean flags indicating if objects are left of center",
        examples=["same_side = objects_left_of_center[0] == is_left_of_center"],
    ),
    "objects_location": ParameterDefinition(
        name="objects_location",
        type=ParameterType.LIST_TUPLE,
        description="List of object locations as (x,y) coordinates",
        examples=["obj_x, obj_y = objects_location[0]"],
    ),
    "objects_speed": ParameterDefinition(
        name="objects_speed",
        type=ParameterType.LIST_FLOAT,
        description="List of objects' speeds in meters per second",
        units="m/s",
        examples=["relative_speed = speed - objects_speed[0]"],
    ),
}
