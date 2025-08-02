from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.rewards.parameters.deepracer_parameters import DeepRacerParameters
from deepracer_research.rewards.template_loader import RewardTemplateLoader


@dataclass
class RewardFunctionBuildConfig:
    """Configuration for reward function building.

    Parameters
    ----------
    scenario : ExperimentalScenario
        The experimental scenario to build for
    parameters : Dict[str, Any], optional
        Custom parameters for the reward function, by default empty dict
    optimization_level : str, optional
        Code optimization level ('none', 'basic', 'advanced'), by default 'basic'
    include_comments : bool, optional
        Whether to include comments in generated code, by default True
    minify_code : bool, optional
        Whether to minify the generated code, by default False
    """

    scenario: ExperimentalScenario
    parameters: Dict[str, Any] = field(default_factory=dict)
    optimization_level: str = "basic"
    include_comments: bool = True
    minify_code: bool = False

    def __post_init__(self):
        """Validate configuration after initialization."""
        valid_optimizations = ["none", "basic", "advanced"]
        if self.optimization_level not in valid_optimizations:
            raise ValueError(
                f"Invalid optimization_level '{self.optimization_level}'. " f"Valid options: {valid_optimizations}"
            )


class RewardFunctionBuilder:
    """Builder for creating deployable reward functions.

    This class provides a systematic way to build reward functions
    with proper code generation, optimization, and deployment preparation.
    """

    def __init__(self):
        """Initialize the reward function builder."""
        self._config: Optional[RewardFunctionBuildConfig] = None
        self._scenario: Optional[ExperimentalScenario] = None
        self._parameters: Dict[str, Any] = {}
        self._optimization_level: str = "basic"
        self._include_comments: bool = True
        self._minify_code: bool = False
        self._validate_parameters: bool = True
        self._template_loader = RewardTemplateLoader()

    def with_scenario(self, scenario: ExperimentalScenario) -> "RewardFunctionBuilder":
        """Set the experimental scenario for the reward function.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The scenario to build the reward function for

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._scenario = scenario
        return self

    def with_parameter_validation(self, validate: bool = True) -> "RewardFunctionBuilder":
        """Set whether to validate parameters against DeepRacer specification.

        Parameters
        ----------
        validate : bool, optional
            Whether to validate parameters, by default True

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._validate_parameters = validate
        return self

    def with_scenario(self, scenario: ExperimentalScenario) -> "RewardFunctionBuilder":
        """Set the experimental scenario for the reward function.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The scenario to build the reward function for

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._scenario = scenario
        return self

    def with_parameters(self, parameters: Dict[str, Any]) -> "RewardFunctionBuilder":
        """Set custom parameters for the reward function.

        Parameters
        ----------
        parameters : Dict[str, Any]
            Custom parameters to use in the reward function

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._parameters.update(parameters)
        return self

    def with_parameter(self, key: str, value: Any) -> "RewardFunctionBuilder":
        """Set a single parameter for the reward function.

        Parameters
        ----------
        key : str
            Parameter name
        value : Any
            Parameter value

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._parameters[key] = value
        return self

    def with_optimization(self, level: str) -> "RewardFunctionBuilder":
        """Set the code optimization level.

        Parameters
        ----------
        level : str
            Optimization level ('none', 'basic', 'advanced')

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._optimization_level = level
        return self

    def with_comments(self, include: bool = True) -> "RewardFunctionBuilder":
        """Set whether to include comments in generated code.

        Parameters
        ----------
        include : bool, optional
            Whether to include comments, by default True

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._include_comments = include
        return self

    def with_minification(self, minify: bool = True) -> "RewardFunctionBuilder":
        """Set whether to minify the generated code.

        Parameters
        ----------
        minify : bool, optional
            Whether to minify the code, by default True

        Returns
        -------
        RewardFunctionBuilder
            Self for method chaining
        """
        self._minify_code = minify
        return self

    def validate_scenario_parameters(self) -> List[str]:
        """Validate parameters for the current scenario.

        Returns
        -------
        List[str]
            List of validation warnings/errors
        """
        if not self._validate_parameters:
            return []

        if self._scenario is None:
            return ["No scenario set for validation"]

        scenario_name = self._scenario.value
        expected_params = DeepRacerParameters.get_parameters_by_scenario(scenario_name)

        warnings = []

        for param_name, param_value in self._parameters.items():
            if param_name.startswith("rewards.") or param_name.startswith("parameters."):
                continue

            if param_name not in expected_params:
                warnings.append(f"Parameter '{param_name}' not in DeepRacer specification")

        return warnings

    def get_available_parameters(self) -> Dict[str, str]:
        """Get available parameters for the current scenario.

        Returns
        -------
        Dict[str, str]
            Dictionary mapping parameter names to descriptions
        """
        if self._scenario is None:
            return {}

        scenario_name = self._scenario.value
        expected_params = DeepRacerParameters.get_parameters_by_scenario(scenario_name)

        return {name: param.description for name, param in expected_params.items()}

    def generate_parameter_documentation(self) -> str:
        """Generate parameter documentation for the current scenario.

        Returns
        -------
        str
            Documentation string for available parameters
        """
        if self._scenario is None:
            return "No scenario selected"

        scenario_name = self._scenario.value
        return DeepRacerParameters.generate_params_comment(scenario_name)

    def build_config(self) -> RewardFunctionBuildConfig:
        """Build the configuration object.

        Returns
        -------
        RewardFunctionBuildConfig
            The built configuration

        Raises
        ------
        ValueError
            If required configuration is missing
        """
        if self._scenario is None:
            raise ValueError("Scenario must be set before building config")

        if self._validate_parameters:
            warnings = self.validate_scenario_parameters()
            if warnings:
                print("Parameter validation warnings:")
                for warning in warnings:
                    print(f"  - {warning}")

        return RewardFunctionBuildConfig(
            scenario=self._scenario,
            parameters=self._parameters.copy(),
            optimization_level=self._optimization_level,
            include_comments=self._include_comments,
            minify_code=self._minify_code,
        )

    def build_function_code(self) -> str:
        """Build the deployable reward function code using YAML templates.

        Returns
        -------
        str
            The complete reward function code ready for deployment
        """
        config = self.build_config()

        from deepracer_research.rewards.template_loader import render_reward_function

        try:
            source_code = render_reward_function(
                scenario=config.scenario, custom_parameters=config.parameters, experiment_id=None
            )
        except Exception as e:
            try:
                source_code = render_reward_function(
                    scenario=ExperimentalScenario.BASIC_FALLBACK, custom_parameters=config.parameters, experiment_id=None
                )
            except Exception as fallback_error:
                raise RuntimeError(
                    f"Failed to generate reward function: {e}. " f"Fallback template also failed: {fallback_error}"
                )

        if config.optimization_level != "none":
            source_code = self._optimize_code(source_code, config)

        if config.minify_code:
            source_code = self._minify_code_string(source_code)

        return source_code

    def build_optimized_function(self) -> str:
        """Build an optimized version of the reward function.

        Returns
        -------
        str
            Optimized reward function code
        """
        return self.with_optimization("advanced").with_minification(True).build_function_code()

    def save_function_code(self, output_path: Union[str, Path]) -> Path:
        """Save the built function code to a file.

        Parameters
        ----------
        output_path : Union[str, Path]
            Path to save the function code

        Returns
        -------
        Path
            The path where the code was saved
        """
        code = self.build_function_code()
        output_path = Path(output_path)

        with open(output_path, "w") as f:
            f.write(code)

        return output_path

    def _optimize_code(self, code: str, config: RewardFunctionBuildConfig) -> str:
        """Apply code optimizations.

        Parameters
        ----------
        code : str
            The source code to optimize
        config : RewardFunctionBuildConfig
            The build configuration

        Returns
        -------
        str
            Optimized code
        """
        optimized_code = code

        if config.optimization_level == "basic":
            optimized_code = self._basic_optimization(optimized_code)
        elif config.optimization_level == "advanced":
            optimized_code = self._advanced_optimization(optimized_code)

        return optimized_code

    def _basic_optimization(self, code: str) -> str:
        """Apply basic code optimizations.

        Parameters
        ----------
        code : str
            Code to optimize

        Returns
        -------
        str
            Optimized code
        """
        lines = code.split("\n")
        optimized_lines = []

        for line in lines:
            if line.strip():
                optimized_lines.append(line.rstrip())
            elif not optimized_lines or optimized_lines[-1].strip():
                optimized_lines.append("")

        return "\n".join(optimized_lines)

    def _advanced_optimization(self, code: str) -> str:
        """Apply advanced code optimizations.

        Parameters
        ----------
        code : str
            Code to optimize

        Returns
        -------
        str
            Optimized code
        """
        optimized_code = self._basic_optimization(code)

        return optimized_code

    def _minify_code_string(self, code: str) -> str:
        """Minify the code string.

        Parameters
        ----------
        code : str
            Code to minify

        Returns
        -------
        str
            Minified code
        """
        lines = code.split("\n")
        minified_lines = []

        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                minified_lines.append(stripped)

        return "\n".join(minified_lines)

    def create_comprehensive_template(self, scenario: ExperimentalScenario) -> str:
        """Create a comprehensive template showing all available parameters.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The scenario to create template for

        Returns
        -------
        str
            Template code with comprehensive parameter usage
        """
        try:
            template = self._template_loader.load_template("comprehensive_example")
            return template.render(self._parameters)
        except (FileNotFoundError, ValueError):
            pass

        try:
            template = self._template_loader.load_template_for_scenario(scenario)
            return template.render(self._parameters)
        except FileNotFoundError:
            if "object" in scenario.value.lower() or "avoidance" in scenario.value.lower():
                try:
                    template = self._template_loader.load_template(ExperimentalScenario.OBJECT_AVOIDANCE)
                    return template.render(self._parameters)
                except FileNotFoundError:
                    pass
        except Exception as e:
            print(f"Warning: Could not load template for {scenario.value}: {e}")

        return self._create_fallback_comprehensive_template(scenario)

    def _create_fallback_comprehensive_template(self, scenario: ExperimentalScenario) -> str:
        """Create a fallback comprehensive template using the basic fallback YAML template."""
        try:
            template = self._template_loader.load_template(ExperimentalScenario.BASIC_FALLBACK)
            return template.render(self._parameters)
        except (FileNotFoundError, ValueError) as e:
            raise RuntimeError(
                f"Cannot create fallback template for scenario {scenario.value}. "
                f"Basic fallback template 'basic_fallback.yaml' is missing: {e}. "
                "Please ensure all required YAML templates are present in the templates directory."
            )

    def get_parameter_groups_for_scenario(self, scenario: ExperimentalScenario) -> Dict[str, List[str]]:
        """Get parameter groups relevant to a specific scenario.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The scenario to get parameter groups for

        Returns
        -------
        Dict[str, List[str]]
            Dictionary mapping group names to parameter lists
        """
        groups = DeepRacerParameters.get_parameter_groups()
        params = DeepRacerParameters.get_parameters_by_scenario(scenario.value)

        filtered_groups = {}
        for group_name, param_names in groups.items():
            relevant_params = [name for name in param_names if name in params]
            if relevant_params:
                filtered_groups[group_name] = relevant_params

        return filtered_groups

    @classmethod
    def create_for_scenario(cls, scenario: ExperimentalScenario, **kwargs) -> "RewardFunctionBuilder":
        """Create a builder configured for a specific scenario.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The scenario to configure for
        **kwargs
            Additional configuration parameters

        Returns
        -------
        RewardFunctionBuilder
            Configured builder instance
        """
        builder = cls().with_scenario(scenario)

        if kwargs:
            builder = builder.with_parameters(kwargs)

        return builder

    @classmethod
    def create_optimized(cls, scenario: ExperimentalScenario, **kwargs) -> "RewardFunctionBuilder":
        """Create an optimized builder for a scenario.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The scenario to configure for
        **kwargs
            Additional configuration parameters

        Returns
        -------
        RewardFunctionBuilder
            Optimized builder instance
        """
        return cls.create_for_scenario(scenario, **kwargs).with_optimization("advanced").with_minification(True)
