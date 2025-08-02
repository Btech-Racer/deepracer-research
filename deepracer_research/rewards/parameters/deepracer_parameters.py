from typing import Any, Dict, List

from deepracer_research.rewards.parameters.core_parameters import CORE_PARAMETERS
from deepracer_research.rewards.parameters.object_parameters import OBJECT_PARAMETERS
from deepracer_research.rewards.parameters.parameter_definition import ParameterDefinition
from deepracer_research.rewards.parameters.status_parameters import STATUS_PARAMETERS
from deepracer_research.rewards.parameters.waypoint_parameters import WAYPOINT_PARAMETERS


class DeepRacerParameters:
    """Comprehensive DeepRacer parameters definitions"""

    @classmethod
    def get_all_parameters(cls) -> Dict[str, ParameterDefinition]:
        """Get all parameter definitions.

        Returns
        -------
        Dict[str, ParameterDefinition]
            Dictionary of all parameter definitions
        """
        all_params = {}
        all_params.update(CORE_PARAMETERS)
        all_params.update(WAYPOINT_PARAMETERS)
        all_params.update(OBJECT_PARAMETERS)
        all_params.update(STATUS_PARAMETERS)
        return all_params

    @classmethod
    def get_parameters_by_scenario(cls, scenario: str) -> Dict[str, ParameterDefinition]:
        """Get parameters relevant to a specific scenario.

        Parameters
        ----------
        scenario : str
            The scenario name (e.g., 'object_avoidance', 'time_trial')

        Returns
        -------
        Dict[str, ParameterDefinition]
            Dictionary of relevant parameter definitions
        """
        params = CORE_PARAMETERS.copy()
        params.update(WAYPOINT_PARAMETERS)
        params.update(STATUS_PARAMETERS)

        if "object" in scenario.lower() or "avoidance" in scenario.lower():
            params.update(OBJECT_PARAMETERS)

        return params

    @classmethod
    def get_parameter_groups(cls) -> Dict[str, List[str]]:
        """Get parameters organized by functional groups.

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping group names to parameter lists
        """
        return {
            "position": ["x", "y", "distance_from_center", "is_left_of_center"],
            "movement": ["speed", "steering_angle", "heading"],
            "track": ["track_width", "track_length", "waypoints", "closest_waypoints"],
            "progress": ["progress", "steps"],
            "status": ["all_wheels_on_track", "is_crashed", "is_offtrack", "is_reversed"],
            "objects": [
                "closest_objects",
                "objects_distance",
                "objects_heading",
                "objects_left_of_center",
                "objects_location",
                "objects_speed",
            ],
        }

    @classmethod
    def validate_params_dict(cls, params: Dict[str, Any]) -> List[str]:
        """Validate a parameters dictionary.

        Parameters
        ----------
        params : Dict[str, Any]
            Parameters dictionary to validate

        Returns
        -------
        List[str]
            List of validation warnings/errors
        """
        warnings = []
        all_params = cls.get_all_parameters()

        for param_name in params.keys():
            if param_name not in all_params:
                warnings.append(f"Unknown parameter: {param_name}")

        core_missing = []
        for param_name in ["all_wheels_on_track", "progress", "speed"]:
            if param_name not in params:
                core_missing.append(param_name)

        if core_missing:
            warnings.append(f"Missing core parameters: {core_missing}")

        return warnings

    @classmethod
    def generate_params_comment(cls, scenario: str = "general") -> str:
        """Generate documentation comment for reward function parameters.

        Parameters
        ----------
        scenario : str, optional
            Scenario to generate comments for, by default "general"

        Returns
        -------
        str
            Multi-line comment documenting available parameters
        """
        params = cls.get_parameters_by_scenario(scenario)
        groups = cls.get_parameter_groups()

        comment_lines = ["'''", "Available parameters from AWS DeepRacer:", ""]

        for group_name, param_names in groups.items():
            relevant_params = [name for name in param_names if name in params]
            if relevant_params:
                comment_lines.append(f"{group_name.title()} Parameters:")
                for param_name in relevant_params:
                    param_def = params[param_name]
                    type_str = param_def.type.value
                    units_str = f" ({param_def.units})" if param_def.units else ""
                    range_str = f" [{param_def.range_info}]" if param_def.range_info else ""
                    comment_lines.append(f"- {param_name} ({type_str}){units_str}{range_str}: {param_def.description}")
                comment_lines.append("")

        comment_lines.append("'''")
        return "\n".join(comment_lines)
