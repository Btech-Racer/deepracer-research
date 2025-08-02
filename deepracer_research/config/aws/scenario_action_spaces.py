from dataclasses import dataclass

from deepracer_research.config.aws.action_space_config import ActionSpaceConfig
from deepracer_research.config.aws.types.action_space_type import ActionSpaceType
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario


@dataclass
class ScenarioActionSpaceConfig:
    """Enhanced action space configurations optimized for different racing scenarios"""

    @classmethod
    def get_for_scenario(cls, scenario: ExperimentalScenario) -> ActionSpaceConfig:
        """Get optimized action space configuration for a specific scenario.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The racing scenario to optimize for

        Returns
        -------
        ActionSpaceConfig
            Optimized action space configuration for the scenario
        """
        scenario_configs = {
            ExperimentalScenario.TIME_TRIAL: cls._get_time_trial_config(),
            ExperimentalScenario.OBJECT_AVOIDANCE: cls._get_object_avoidance_config(),
            ExperimentalScenario.HEAD_TO_HEAD: cls._get_head_to_head_config(),
            ExperimentalScenario.CENTERLINE_FOLLOWING: cls._get_centerline_following_config(),
            ExperimentalScenario.SPEED_OPTIMIZATION: cls._get_speed_optimization_config(),
        }

        return scenario_configs.get(scenario, cls._get_default_config())

    @classmethod
    def _get_time_trial_config(cls) -> ActionSpaceConfig:
        """Optimized action space for time trial racing"""
        return ActionSpaceConfig(
            type=ActionSpaceType.DISCRETE,
            num_speed_levels=4,
            num_steering_levels=7,
            discrete_actions={
                "speed_levels": [1.5, 2.5, 3.5, 4.5],
                "steering_angles": [-30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0],
            },
        )

    @classmethod
    def _get_object_avoidance_config(cls) -> ActionSpaceConfig:
        """Optimized action space for object avoidance scenarios"""
        return ActionSpaceConfig(
            type=ActionSpaceType.DISCRETE,
            num_speed_levels=3,
            num_steering_levels=5,
            discrete_actions={"speed_levels": [1.0, 2.0, 3.0], "steering_angles": [-25.0, -12.5, 0.0, 12.5, 25.0]},
        )

    @classmethod
    def _get_head_to_head_config(cls) -> ActionSpaceConfig:
        """Optimized action space for head-to-head competitive racing"""
        return ActionSpaceConfig(
            type=ActionSpaceType.DISCRETE,
            num_speed_levels=5,
            num_steering_levels=9,
            discrete_actions={
                "speed_levels": [1.0, 2.0, 3.0, 4.0, 5.0],
                "steering_angles": [-30.0, -22.5, -15.0, -7.5, 0.0, 7.5, 15.0, 22.5, 30.0],
            },
        )

    @classmethod
    def _get_centerline_following_config(cls) -> ActionSpaceConfig:
        """Optimized action space for centerline following (beginner-friendly)"""
        return ActionSpaceConfig(
            type=ActionSpaceType.DISCRETE,
            num_speed_levels=3,
            num_steering_levels=5,
            discrete_actions={"speed_levels": [1.2, 2.0, 2.8], "steering_angles": [-20.0, -10.0, 0.0, 10.0, 20.0]},
        )

    @classmethod
    def _get_speed_optimization_config(cls) -> ActionSpaceConfig:
        """Optimized action space for speed optimization scenarios"""
        return ActionSpaceConfig(
            type=ActionSpaceType.DISCRETE,
            num_speed_levels=6,
            num_steering_levels=5,
            discrete_actions={
                "speed_levels": [2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
                "steering_angles": [-25.0, -12.0, 0.0, 12.0, 25.0],
            },
        )

    @classmethod
    def _get_default_config(cls) -> ActionSpaceConfig:
        """Default balanced action space configuration"""
        return ActionSpaceConfig(
            type=ActionSpaceType.DISCRETE,
            num_speed_levels=3,
            num_steering_levels=5,
            discrete_actions={"speed_levels": [1.0, 2.5, 4.0], "steering_angles": [-30.0, -15.0, 0.0, 15.0, 30.0]},
        )

    @classmethod
    def get_continuous_for_scenario(cls, scenario: ExperimentalScenario) -> ActionSpaceConfig:
        """Get continuous action space optimized for a specific scenario

        Parameters
        ----------
        scenario : ExperimentalScenario
            The racing scenario to optimize for

        Returns
        -------
        ActionSpaceConfig
            Continuous action space configuration for the scenario
        """
        scenario_continuous_configs = {
            ExperimentalScenario.TIME_TRIAL: ActionSpaceConfig(
                type=ActionSpaceType.CONTINUOUS,
                speed_range={"min": 1.5, "max": 5.0},
                steering_range={"min": -30.0, "max": 30.0},
            ),
            ExperimentalScenario.OBJECT_AVOIDANCE: ActionSpaceConfig(
                type=ActionSpaceType.CONTINUOUS,
                speed_range={"min": 0.8, "max": 3.0},
                steering_range={"min": -25.0, "max": 25.0},
            ),
            ExperimentalScenario.HEAD_TO_HEAD: ActionSpaceConfig(
                type=ActionSpaceType.CONTINUOUS,
                speed_range={"min": 1.0, "max": 5.0},
                steering_range={"min": -30.0, "max": 30.0},
            ),
            ExperimentalScenario.CENTERLINE_FOLLOWING: ActionSpaceConfig(
                type=ActionSpaceType.CONTINUOUS,
                speed_range={"min": 1.0, "max": 3.0},
                steering_range={"min": -20.0, "max": 20.0},
            ),
            ExperimentalScenario.SPEED_OPTIMIZATION: ActionSpaceConfig(
                type=ActionSpaceType.CONTINUOUS,
                speed_range={"min": 2.0, "max": 5.0},
                steering_range={"min": -25.0, "max": 25.0},
            ),
        }

        return scenario_continuous_configs.get(scenario, ActionSpaceConfig())

    @classmethod
    def get_scenario_description(cls, scenario: ExperimentalScenario) -> str:
        """Get description of the action space optimization for a scenario

        Parameters
        ----------
        scenario : ExperimentalScenario
            The racing scenario

        Returns
        -------
        str
            Description of the action space optimization strategy
        """
        descriptions = {
            ExperimentalScenario.TIME_TRIAL: "Aggressive speeds (1.5-4.5 m/s) with fine steering control for optimal racing lines",
            ExperimentalScenario.OBJECT_AVOIDANCE: "Conservative speeds (1.0-3.0 m/s) with responsive steering for obstacle navigation",
            ExperimentalScenario.HEAD_TO_HEAD: "Full speed range (1.0-5.0 m/s) with precise steering for competitive positioning",
            ExperimentalScenario.CENTERLINE_FOLLOWING: "Beginner-friendly speeds (1.2-2.8 m/s) with gentle steering for stability",
            ExperimentalScenario.SPEED_OPTIMIZATION: "High-speed focus (2.0-4.5 m/s) with efficient steering for speed optimization",
        }

        return descriptions.get(scenario, "Balanced action space for general racing scenarios")
