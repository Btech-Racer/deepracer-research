from dataclasses import dataclass
from typing import List, Optional

from deepracer_research.rewards.parameters.parameter_type import ParameterType


@dataclass
class ParameterDefinition:
    """Definition of a DeepRacer parameter.

    Parameters
    ----------
    name : str
        Parameter name
    type : ParameterType
        Data type of the parameter
    description : str
        Description of what the parameter represents
    units : Optional[str]
        Units of measurement (if applicable)
    range_info : Optional[str]
        Information about value ranges
    examples : Optional[List[str]]
        Example usage patterns
    """

    name: str
    type: ParameterType
    description: str
    units: Optional[str] = None
    range_info: Optional[str] = None
    examples: Optional[List[str]] = None
