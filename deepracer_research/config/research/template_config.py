from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario

SCENARIO_TEMPLATE_MAPPING = {
    ExperimentalScenario.CENTERLINE_FOLLOWING: "centerline_following",
    ExperimentalScenario.SPEED_OPTIMIZATION: "speed_optimization",
    ExperimentalScenario.OBJECT_AVOIDANCE: "object_avoidance",
    ExperimentalScenario.OBJECT_AVOIDANCE_STATIC: "object_avoidance_static",
    ExperimentalScenario.OBJECT_AVOIDANCE_DYNAMIC: "object_avoidance_dynamic",
    ExperimentalScenario.TIME_TRIAL: "time_trial",
    ExperimentalScenario.HEAD_TO_HEAD: "head_to_head",
    ExperimentalScenario.BASIC_FALLBACK: "basic_fallback",
}

RACING_SCENARIO_TEMPLATES = {
    "centerline_following": "centerline_following",
    "speed_optimization": "speed_optimization",
    "obstacle_avoidance": "object_avoidance",
    "object_avoidance": "object_avoidance",
    "time_trial": "time_trial",
    "head_to_head": "head_to_head",
    "multi_agent": "object_avoidance_dynamic",
    "continuous_control": "centerline_following",
}

RACING_SCENARIO_TO_EXPERIMENTAL = {
    "centerline_following": ExperimentalScenario.CENTERLINE_FOLLOWING,
    "speed_optimization": ExperimentalScenario.SPEED_OPTIMIZATION,
    "obstacle_avoidance": ExperimentalScenario.OBJECT_AVOIDANCE,
    "object_avoidance": ExperimentalScenario.OBJECT_AVOIDANCE,
    "time_trial": ExperimentalScenario.TIME_TRIAL,
    "head_to_head": ExperimentalScenario.HEAD_TO_HEAD,
    "multi_agent": ExperimentalScenario.OBJECT_AVOIDANCE_DYNAMIC,
    "continuous_control": ExperimentalScenario.CENTERLINE_FOLLOWING,
}

DEFAULT_TEMPLATE = "basic_fallback"


def get_template_for_scenario(scenario: ExperimentalScenario) -> str:
    """Get template name for experimental scenario.

    Parameters
    ----------
    scenario : ExperimentalScenario
        The experimental scenario

    Returns
    -------
    str
        Template name to use
    """
    return SCENARIO_TEMPLATE_MAPPING.get(scenario, DEFAULT_TEMPLATE)


def get_template_for_racing_scenario(scenario_name: str) -> str:
    """Get template name for racing scenario.

    Parameters
    ----------
    scenario_name : str
        Name of the racing scenario

    Returns
    -------
    str
        Template name to use
    """
    return RACING_SCENARIO_TEMPLATES.get(scenario_name, DEFAULT_TEMPLATE)


def get_experimental_scenario_for_racing(scenario_name: str) -> ExperimentalScenario:
    """Get ExperimentalScenario enum for racing scenario name.

    Parameters
    ----------
    scenario_name : str
        Name of the racing scenario

    Returns
    -------
    ExperimentalScenario
        Corresponding experimental scenario enum
    """
    return RACING_SCENARIO_TO_EXPERIMENTAL.get(scenario_name, ExperimentalScenario.BASIC_FALLBACK)


def list_supported_scenarios() -> list:
    """List all supported experimental scenarios.

    Returns
    -------
    list
        List of supported scenario enum values
    """
    return list(SCENARIO_TEMPLATE_MAPPING.keys())


def list_supported_racing_scenarios() -> list:
    """List all supported racing scenario names.

    Returns
    -------
    list
        List of supported racing scenario names
    """
    return list(RACING_SCENARIO_TEMPLATES.keys())
