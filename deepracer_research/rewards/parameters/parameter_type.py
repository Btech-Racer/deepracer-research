from enum import Enum


class ParameterType(Enum):
    """Types of parameters available in DeepRacer reward functions."""

    BOOLEAN = "boolean"
    FLOAT = "float"
    INT = "int"
    LIST_FLOAT = "list_float"
    LIST_BOOLEAN = "list_boolean"
    LIST_TUPLE = "list_tuple"
    LIST_INT = "list_int"
