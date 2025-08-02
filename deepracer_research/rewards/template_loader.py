from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from deepracer_research.config.research.template_config import (
    get_experimental_scenario_for_racing,
    get_template_for_racing_scenario,
    get_template_for_scenario,
)
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario


@dataclass
class RewardTemplate:
    """Represents a reward function template."""

    name: str
    description: str
    scenario: str
    metadata: Dict[str, Any]
    parameters: Dict[str, Any]
    template: str

    @classmethod
    def from_yaml(cls, yaml_content: str) -> "RewardTemplate":
        """Create RewardTemplate from YAML content."""
        data = yaml.safe_load(yaml_content)
        return cls(
            name=data["name"],
            description=data["description"],
            scenario=data["scenario"],
            metadata=data.get("metadata", {}),
            parameters=data.get("parameters", {}),
            template=data["template"],
        )

    def render(self, custom_parameters: Optional[Dict[str, Any]] = None) -> str:
        """Render the template with parameters.

        Parameters
        ----------
        custom_parameters : Dict[str, Any], optional
            Custom parameters to override defaults

        Returns
        -------
        str
            Rendered reward function code
        """
        render_params = self.parameters.copy()

        if custom_parameters:
            render_params = self._deep_merge(render_params, custom_parameters)

        rendered_code = self.template

        rendered_code = self._replace_parameters(rendered_code, render_params)

        return rendered_code

    def _replace_parameters(self, template: str, parameters: Dict[str, Any]) -> str:
        """Replace parameter placeholders in template.

        Parameters
        ----------
        template : str
            Template string with {{ parameters.key }} placeholders
        parameters : Dict[str, Any]
            Parameters to substitute

        Returns
        -------
        str
            Template with parameters substituted
        """
        result = template

        def replace_recursive(obj, prefix="parameters"):
            nonlocal result
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{prefix}.{key}"
                    if isinstance(value, dict):
                        replace_recursive(value, current_path)
                    else:
                        placeholder_no_spaces = f"{{{{{current_path}}}}}"
                        placeholder_with_spaces = f"{{{{ {current_path} }}}}"
                        result = result.replace(placeholder_no_spaces, str(value))
                        result = result.replace(placeholder_with_spaces, str(value))

        replace_recursive(parameters)

        return result

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        return result


class RewardTemplateLoader:
    """Loads and manages reward function templates."""

    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize the template loader.

        Parameters
        ----------
        templates_dir : Path, optional
            Directory containing template files. If None, uses default.
        """
        if templates_dir is None:
            templates_dir = Path(__file__).parent / "templates"

        self.templates_dir = Path(templates_dir)
        self._template_cache: Dict[str, RewardTemplate] = {}

    def load_template(self, template_name: str) -> RewardTemplate:
        """Load a specific template by name.

        Parameters
        ----------
        template_name : str
            Name of the template file (without .yaml extension)

        Returns
        -------
        RewardTemplate
            The loaded template

        Raises
        ------
        FileNotFoundError
            If template file doesn't exist
        ValueError
            If template content is invalid
        """
        if template_name in self._template_cache:
            return self._template_cache[template_name]

        template_path = self.templates_dir / f"{template_name}.yaml"
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        try:
            with open(template_path, "r") as f:
                template_content = f.read()

            template = RewardTemplate.from_yaml(template_content)
            self._template_cache[template_name] = template
            return template

        except Exception as e:
            raise ValueError(f"Failed to load template {template_name}: {e}")

    def load_template_for_scenario(self, scenario: ExperimentalScenario) -> RewardTemplate:
        """Load template based on experimental scenario using configuration mapping.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The experimental scenario

        Returns
        -------
        RewardTemplate
            The loaded template
        """
        template_name = get_template_for_scenario(scenario)
        return self.load_template(template_name)

    def load_template_for_racing_scenario(self, scenario_name: str) -> RewardTemplate:
        """Load template based on racing scenario name using configuration mapping.

        Parameters
        ----------
        scenario_name : str
            The racing scenario name

        Returns
        -------
        RewardTemplate
            The loaded template
        """
        template_name = get_template_for_racing_scenario(scenario_name)
        return self.load_template(template_name)

    def list_available_templates(self) -> List[str]:
        """List all available template names.

        Returns
        -------
        List[str]
            List of available template names
        """
        if not self.templates_dir.exists():
            return []

        return [f.stem for f in self.templates_dir.glob("*.yaml") if f.is_file()]

    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get metadata about a template without loading it fully.

        Parameters
        ----------
        template_name : str
            Name of the template

        Returns
        -------
        Dict[str, Any]
            Template metadata
        """
        template = self.load_template(template_name)
        return {
            "name": template.name,
            "description": template.description,
            "scenario": template.scenario,
            "metadata": template.metadata,
            "parameters": list(template.parameters.keys()),
        }


class RewardFunctionRenderer:
    """Renders reward functions from templates with custom parameters."""

    def __init__(self, loader: Optional[RewardTemplateLoader] = None):
        """Initialize the renderer.

        Parameters
        ----------
        loader : RewardTemplateLoader, optional
            Template loader. If None, creates default loader.
        """
        self.loader = loader or RewardTemplateLoader()

    def render_for_scenario(
        self,
        scenario: ExperimentalScenario,
        custom_parameters: Optional[Dict[str, Any]] = None,
        experiment_id: Optional[str] = None,
    ) -> str:
        """Render reward function for experimental scenario.

        Parameters
        ----------
        scenario : ExperimentalScenario
            The experimental scenario
        custom_parameters : Dict[str, Any], optional
            Custom parameters to override template defaults
        experiment_id : str, optional
            Experiment ID to include in comments

        Returns
        -------
        str
            Rendered reward function code
        """
        template = self.loader.load_template_for_scenario(scenario)
        code = template.render(custom_parameters)

        if experiment_id:
            code = self._add_experiment_metadata(code, experiment_id, scenario)

        return code

    def render_for_racing_scenario(
        self, scenario_name: str, custom_parameters: Optional[Dict[str, Any]] = None, experiment_id: Optional[str] = None
    ) -> str:
        """Render reward function for racing scenario with optimized parameters.

        Parameters
        ----------
        scenario_name : str
            The racing scenario name
        custom_parameters : Dict[str, Any], optional
            Custom parameters to override template defaults
        experiment_id : str, optional
            Experiment ID to include in comments

        Returns
        -------
        str
            Rendered reward function code
        """
        # Get the corresponding experimental scenario for parameter optimization
        experimental_scenario = get_experimental_scenario_for_racing(scenario_name)

        # Load the template for this racing scenario
        template = self.loader.load_template_for_racing_scenario(scenario_name)

        # Get scenario-specific optimized parameters from AWSDeepRacerConfig
        from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig

        optimized_params = AWSDeepRacerConfig._get_scenario_specific_parameters(experimental_scenario)

        # Merge optimized parameters with custom parameters (custom takes precedence)
        final_parameters = {}
        if optimized_params:
            final_parameters.update(optimized_params)
        if custom_parameters:
            final_parameters.update(custom_parameters)

        # Render with the merged parameters
        code = template.render(final_parameters if final_parameters else None)

        if experiment_id:
            code = self._add_experiment_metadata(code, experiment_id, experimental_scenario)

        return code

    def render_from_template(
        self, template_name: str, custom_parameters: Optional[Dict[str, Any]] = None, experiment_id: Optional[str] = None
    ) -> str:
        """Render reward function from specific template.

        Parameters
        ----------
        template_name : str
            Name of the template to use
        custom_parameters : Dict[str, Any], optional
            Custom parameters to override template defaults
        experiment_id : str, optional
            Experiment ID to include in comments

        Returns
        -------
        str
            Rendered reward function code
        """
        template = self.loader.load_template(template_name)
        code = template.render(custom_parameters)

        if experiment_id:
            scenario = ExperimentalScenario(template.scenario)
            code = self._add_experiment_metadata(code, experiment_id, scenario)

        return code

    def _add_experiment_metadata(self, code: str, experiment_id: str, scenario: ExperimentalScenario) -> str:
        """Add experiment metadata to the generated code.

        Parameters
        ----------
        code : str
            The generated reward function code
        experiment_id : str
            Experiment identifier
        scenario : ExperimentalScenario
            Experimental scenario

        Returns
        -------
        str
            Code with added metadata
        """
        lines = code.split("\n")

        for i, line in enumerate(lines):
            if "'''" in line and i > 0:
                experiment_info = f"""
      Generated for experiment: {experiment_id}
      Scenario: {scenario.value}
      Template: YAML-based reward function
      """
                lines.insert(i, experiment_info)
                break

        return "\n".join(lines)


def render_reward_function(
    scenario: ExperimentalScenario, custom_parameters: Optional[Dict[str, Any]] = None, experiment_id: Optional[str] = None
) -> str:
    """Render reward function for scenario with custom parameters.

    Parameters
    ----------
    scenario : ExperimentalScenario
        The experimental scenario
    custom_parameters : Dict[str, Any], optional
        Custom parameters to override defaults
    experiment_id : str, optional
        Experiment ID for metadata

    Returns
    -------
    str
        Rendered reward function code
    """
    renderer = RewardFunctionRenderer()
    return renderer.render_for_scenario(scenario, custom_parameters, experiment_id)


def render_reward_function_for_racing_scenario(
    scenario_name: str, custom_parameters: Optional[Dict[str, Any]] = None, experiment_id: Optional[str] = None
) -> str:
    """Render reward function for racing scenario with custom parameters.

    Parameters
    ----------
    scenario_name : str
        The racing scenario name
    custom_parameters : Dict[str, Any], optional
        Custom parameters to override defaults
    experiment_id : str, optional
        Experiment ID for metadata

    Returns
    -------
    str
        Rendered reward function code
    """
    renderer = RewardFunctionRenderer()
    return renderer.render_for_racing_scenario(scenario_name, custom_parameters, experiment_id)


def get_available_templates() -> List[str]:
    """Get list of available reward function templates.

    Returns
    -------
    List[str]
        List of available template names
    """
    loader = RewardTemplateLoader()
    return loader.list_available_templates()


def get_template_info(template_name: str) -> Dict[str, Any]:
    """Get information about a specific template.

    Parameters
    ----------
    template_name : str
        Name of the template

    Returns
    -------
    Dict[str, Any]
        Template information and metadata
    """
    loader = RewardTemplateLoader()
    return loader.get_template_info(template_name)
