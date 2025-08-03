import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Union

from deepracer_research.models.build.aws_model_config import AWSModelConfig
from deepracer_research.utils.logger import info


class AWSDeepRacerModel:
    """Represents a complete AWS DeepRacer model ready for deployment.

    Parameters
    ----------
    config : AWSModelConfig
        Model configuration
    reward_function_code : str
        AWS-compatible reward function code
    metadata : Dict[str, Any]
        Complete model metadata
    model_files : Dict[str, Union[str, bytes]]
        Model files dictionary
    """

    def __init__(
        self,
        config: AWSModelConfig,
        reward_function_code: str,
        metadata: Dict[str, Any],
        model_files: Dict[str, Union[str, bytes]],
    ):
        """Initialize the AWS DeepRacer model.

        Parameters
        ----------
        config : AWSModelConfig
            Model configuration
        reward_function_code : str
            AWS-compatible reward function code
        metadata : Dict[str, Any]
            Complete model metadata
        model_files : Dict[str, Union[str, bytes]]
            Model files dictionary

        Returns
        -------
        None
        """
        self.config = config
        self.reward_function_code = reward_function_code
        self.metadata = metadata
        self.model_files = model_files

    def export_to_directory(self, output_path: Union[str, Path]) -> Path:
        """Export the model to a directory structure.

        Parameters
        ----------
        output_path : Union[str, Path]
            Path to export the model to

        Returns
        -------
        Path
            Path to the exported model directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        reward_file = output_path / "reward_function.py"
        reward_file.write_text(self.reward_function_code)

        metadata_file = output_path / "model_metadata.json"
        metadata_file.write_text(json.dumps(self.metadata, indent=2))

        for filename, content in self.model_files.items():
            file_path = output_path / filename
            if isinstance(content, str):
                file_path.write_text(content)
            else:
                file_path.write_bytes(content)

        info(f"Model exported to {output_path}")
        return output_path

    def export_to_zip(self, output_path: Union[str, Path]) -> Path:
        """Export the model as a ZIP file.

        Parameters
        ----------
        output_path : Union[str, Path]
            Path for the output ZIP file

        Returns
        -------
        Path
            Path to the created ZIP file
        """
        output_path = Path(output_path)

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            zip_file.writestr("reward_function.py", self.reward_function_code)

            zip_file.writestr("model_metadata.json", json.dumps(self.metadata, indent=2))

            for filename, content in self.model_files.items():
                if isinstance(content, str):
                    zip_file.writestr(filename, content)
                else:
                    zip_file.writestr(filename, content)

        info(f"Model exported to ZIP: {output_path}")
        return output_path

    def get_reward_function_code(self) -> str:
        """Get the reward function code.

        Returns
        -------
        str
            AWS-compatible reward function code
        """
        return self.reward_function_code

    def get_metadata(self) -> Dict[str, Any]:
        """Get the complete model metadata.

        Returns
        -------
        Dict[str, Any]
            Model metadata dictionary
        """
        return self.metadata.copy()

    def get_config(self) -> AWSModelConfig:
        """Get the model configuration.

        Returns
        -------
        AWSModelConfig
            The model configuration object
        """
        return self.config

    def get_model_files(self) -> Dict[str, Union[str, bytes]]:
        """Get the model files dictionary.

        Returns
        -------
        Dict[str, Union[str, bytes]]
            Copy of the model files dictionary
        """
        return self.model_files.copy()

    def validate_aws_compatibility(self) -> List[str]:
        """Validate AWS DeepRacer compatibility.

        Returns
        -------
        List[str]
            List of validation warnings/issues
        """
        issues = []

        try:
            compile(self.reward_function_code, "<reward_function>", "exec")
        except SyntaxError as e:
            issues.append(f"Reward function syntax error: {e}")

        required_fields = ["format_version", "training_algorithm", "model_metadata"]
        for field in required_fields:
            if field not in self.metadata:
                issues.append(f"Missing required metadata field: {field}")

        action_space = self.metadata.get("model_metadata", {}).get("action_space", {})
        if action_space.get("type") not in ["continuous", "discrete"]:
            issues.append("Invalid or missing action space type")

        model_metadata = self.metadata.get("model_metadata", {})
        if not model_metadata.get("model_name"):
            issues.append("Missing model name in metadata")

        if not model_metadata.get("model_description"):
            issues.append("Missing model description in metadata")

        return issues

    def is_valid_for_deployment(self) -> bool:
        """Check if the model is valid for AWS deployment.

        Returns
        -------
        bool
            True if the model is valid for deployment, False otherwise
        """
        issues = self.validate_aws_compatibility()
        return len(issues) == 0

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the model.

        Returns
        -------
        Dict[str, Any]
            Model summary dictionary
        """
        summary = {
            "name": self.config.model_name,
            "description": self.config.description,
            "version": self.config.version,
            "action_space_type": self.config.action_space_type,
            "reward_scenario": self.metadata.get("model_metadata", {}).get("reward_function", {}).get("scenario", "unknown"),
            "model_files_count": len(self.model_files),
            "is_deployment_ready": self.is_valid_for_deployment(),
            "validation_issues_count": len(self.validate_aws_compatibility()),
        }

        return summary

    def __str__(self) -> str:
        """String representation of the model.

        Returns
        -------
        str
            String representation
        """
        return f"AWSDeepRacerModel(name='{self.config.model_name}', version='{self.config.version}')"

    def __repr__(self) -> str:
        """Detailed string representation of the model.

        Returns
        -------
        str
            Detailed string representation
        """
        scenario = self.metadata.get("model_metadata", {}).get("reward_function", {}).get("scenario", "unknown")
        return (
            f"AWSDeepRacerModel(name='{self.config.model_name}', "
            f"version='{self.config.version}', "
            f"scenario='{scenario}')"
        )
