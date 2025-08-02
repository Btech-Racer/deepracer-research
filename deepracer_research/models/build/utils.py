from typing import Optional

from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.models.build.aws_deepracer_model import AWSDeepRacerModel
from deepracer_research.models.build.aws_model_builder import AWSModelBuilder


def create_aws_model(
    name: str, description: str, scenario: Optional[ExperimentalScenario] = None, **kwargs
) -> AWSDeepRacerModel:
    """Create an AWS DeepRacer model with minimal configuration.

    Parameters
    ----------
    name : str
        Model name
    description : str
        Model description
    scenario : Optional[ExperimentalScenario], optional
        Reward function scenario, by default None
    **kwargs
        Additional configuration options including:
        - version: str - Model version
        - action_space: ActionSpaceType - Action space type
        - hyperparameters: Union[AWSHyperparameters, Dict] - Training hyperparameters
        - metadata: Union[AWSModelMetadata, Dict] - Model metadata
        - sensor_config: Union[SensorConfig, Dict] - Sensor configuration

    Returns
    -------
    AWSDeepRacerModel
        The built AWS DeepRacer model
    """
    builder = AWSModelBuilder().with_name(name).with_description(description)

    if scenario:
        scenario_kwargs = {}
        builder_kwargs = {}

        for key, value in kwargs.items():
            if key in ["version", "action_space", "hyperparameters", "metadata", "sensor_config"]:
                builder_kwargs[key] = value
            else:
                scenario_kwargs[key] = value

        builder = builder.with_reward_scenario(scenario, **scenario_kwargs)
        kwargs = builder_kwargs

    for key, value in kwargs.items():
        if key == "version":
            builder = builder.with_version(value)
        elif key == "action_space":
            builder = builder.with_action_space(value)
        elif key == "hyperparameters":
            builder = builder.with_hyperparameters(value)
        elif key == "metadata":
            builder = builder.with_metadata(value)
        elif key == "sensor_config":
            builder = builder.with_sensor_config(value)

    return builder.build()


def create_simple_aws_model(name: str, description: str = "", **kwargs) -> AWSDeepRacerModel:
    """Create a simple AWS DeepRacer model with default settings.

    Parameters
    ----------
    name : str
        Model name
    description : str, optional
        Model description, by default ""
    **kwargs
        Additional parameters passed to create_aws_model

    Returns
    -------
    AWSDeepRacerModel
        The built AWS DeepRacer model with default settings
    """
    if not description:
        description = f"Simple AWS DeepRacer model: {name}"

    return create_aws_model(name=name, description=description, scenario=ExperimentalScenario.CENTERLINE_FOLLOWING, **kwargs)


def create_research_aws_model(
    name: str, description: str, scenario: ExperimentalScenario, research_parameters: dict, **kwargs
) -> AWSDeepRacerModel:
    """Create an AWS DeepRacer model optimized for research purposes.


    Parameters
    ----------
    name : str
        Model name
    description : str
        Model description
    scenario : ExperimentalScenario
        Research scenario to implement
    research_parameters : dict
        Research-specific parameters for the scenario
    **kwargs
        Additional configuration options

    Returns
    -------
    AWSDeepRacerModel
        The built research-optimized AWS DeepRacer model
    """
    if "metadata" not in kwargs:
        kwargs["metadata"] = {}

    if isinstance(kwargs["metadata"], dict):
        kwargs["metadata"].update(
            {
                "research_experiment": True,
                "research_scenario": scenario.value,
                "research_parameters": research_parameters,
                "optimization_target": "research_validation",
            }
        )

    return create_aws_model(name=name, description=description, scenario=scenario, **research_parameters, **kwargs)
