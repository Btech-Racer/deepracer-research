from typing import Dict

from deepracer_research.rewards.parameters.parameter_definition import ParameterDefinition
from deepracer_research.rewards.parameters.parameter_type import ParameterType

STATUS_PARAMETERS: Dict[str, ParameterDefinition] = {
    "is_crashed": ParameterDefinition(
        name="is_crashed",
        type=ParameterType.BOOLEAN,
        description="Boolean flag to indicate whether the agent has crashed",
        examples=["if is_crashed: return 1e-3"],
    ),
    "is_offtrack": ParameterDefinition(
        name="is_offtrack",
        type=ParameterType.BOOLEAN,
        description="Boolean flag to indicate whether the agent has gone off track",
        examples=["if is_offtrack: return 1e-3"],
    ),
    "is_reversed": ParameterDefinition(
        name="is_reversed",
        type=ParameterType.BOOLEAN,
        description="Flag to indicate if agent is driving clockwise (True) or counter-clockwise (False)",
        examples=["if is_reversed: reward *= 0.1"],
    ),
}
