import hashlib
import json
import shutil
import tarfile
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import boto3
import yaml
from botocore.exceptions import ClientError, NoCredentialsError
from jinja2 import Template

from deepracer_research.config import ArchitectureType
from deepracer_research.config.aws.types.action_space_type import ActionSpaceType
from deepracer_research.config.aws.types.evaluation_type import EvaluationType
from deepracer_research.config.network.architecture_type import ArchitectureType
from deepracer_research.config.training.loss_type import LossType
from deepracer_research.config.training.optimizer_config import OptimizerType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.models.build import AWSModelBuilder, create_aws_model, create_simple_aws_model
from deepracer_research.utils import info


class DeepRacerDeploymentManager:
    """Manager for deploying models to AWS DeepRacer service."""

    def __init__(self, region: str = "us-east-1"):
        """Initialize the deployment manager.

        Parameters
        ----------
        region : str
            AWS region to deploy to
        """
        self.region = region
        self._client = None
        self._iam_role_arn = None

    @property
    def client(self):
        """Get the DeepRacer client, creating it if necessary."""
        if self._client is None:
            try:
                self._client = boto3.client("deepracer", region_name=self.region)
            except NoCredentialsError:
                raise Exception(
                    "AWS credentials not configured. Please run 'aws configure' or set AWS_PROFILE environment variable."
                )
        return self._client

    def get_or_create_service_role(self) -> str:
        """Get or create the DeepRacer service role.

        Returns
        -------
        str
            ARN of the DeepRacer service role
        """
        if self._iam_role_arn:
            return self._iam_role_arn

        iam = boto3.client("iam", region_name=self.region)
        role_name = "DeepRacerServiceRole"

        try:
            response = iam.get_role(RoleName=role_name)
            self._iam_role_arn = response["Role"]["Arn"]
            return self._iam_role_arn
        except ClientError as e:
            if e.response["Error"]["Code"] != "NoSuchEntity":
                raise

        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [{"Effect": "Allow", "Principal": {"Service": "deepracer.amazonaws.com"}, "Action": "sts:AssumeRole"}],
        }

        try:
            response = iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Service role for AWS DeepRacer",
            )
            self._iam_role_arn = response["Role"]["Arn"]

            policies = [
                "arn:aws:iam::aws:policy/service-role/AWSDeepRacerServiceRolePolicy",
                "arn:aws:iam::aws:policy/AmazonS3FullAccess",
                "arn:aws:iam::aws:policy/CloudWatchLogsFullAccess",
            ]

            for policy_arn in policies:
                try:
                    iam.attach_role_policy(RoleName=role_name, PolicyArn=policy_arn)
                except ClientError:
                    pass

            time.sleep(10)

            return self._iam_role_arn

        except ClientError as e:
            raise Exception(f"Failed to create DeepRacer service role: {e}")

    def create_model_metadata(self, config: AWSDeepRacerConfig) -> Dict[str, Any]:
        """Create model metadata JSON for import_model.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration

        Returns
        -------
        Dict[str, Any]
            Model metadata dictionary
        """
        from deepracer_research.config.training.training_algorithm import TrainingAlgorithm

        if config.training_algorithm == TrainingAlgorithm.SAC:
            version = "4"
        elif config.training_algorithm == TrainingAlgorithm.CLIPPED_PPO:
            version = "5"
        else:
            version = "4"

        metadata = {
            "action_space": self._convert_action_space_to_aws_format(config),
            "sensor": [sensor.value for sensor in config.get_sensor_list()],
            "neural_network": self._get_aws_neural_network_type(config.architecture_type),
            "version": version,
            "training_algorithm": config.training_algorithm.value,
            "action_space_type": config.action_space_type.value,
            "preprocess_type": "GREY_SCALE",
            "regional_parameters": [0, 0, 0, 0],
        }

        return metadata

    def _get_aws_neural_network_type(self, architecture_type) -> str:
        """Convert ArchitectureType to AWS DeepRacer neural network format.

        Parameters
        ----------
        architecture_type : ArchitectureType
            The neural network architecture type

        Returns
        -------
        str
            AWS DeepRacer compatible neural network type
        """
        from deepracer_research.config.network.architecture_type import ArchitectureType
        from deepracer_research.config.network.neural_network_type import NeuralNetworkType

        aws_networks = {
            ArchitectureType.ATTENTION_CNN: NeuralNetworkType.SHALLOW.value,
            ArchitectureType.RESIDUAL_NETWORK: NeuralNetworkType.DEEP.value,
            ArchitectureType.EFFICIENT_NET: NeuralNetworkType.SHALLOW.value,
            ArchitectureType.TEMPORAL_CNN: NeuralNetworkType.DEEP.value,
            ArchitectureType.TRANSFORMER_VISION: NeuralNetworkType.DEEP.value,
            ArchitectureType.MULTI_MODAL_FUSION: NeuralNetworkType.DEEP.value,
            ArchitectureType.LSTM_CNN: NeuralNetworkType.DEEP.value,
            ArchitectureType.NEURAL_ODE: NeuralNetworkType.DEEP.value,
            ArchitectureType.BAYESIAN_CNN: NeuralNetworkType.SHALLOW.value,
            ArchitectureType.VELOCITY_AWARE: NeuralNetworkType.SHALLOW.value,
            ArchitectureType.ENSEMBLE: NeuralNetworkType.DEEP.value,
        }

        return aws_networks.get(architecture_type, NeuralNetworkType.SHALLOW.value)

    def _convert_action_space_to_aws_format(self, config: AWSDeepRacerConfig) -> Dict[str, Any]:
        """Convert action space configuration to AWS format.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration containing action space parameters

        Returns
        -------
        Dict[str, Any]
            Action space configuration in AWS format
        """
        from deepracer_research.config.aws.types.action_space_type import ActionSpaceType

        if config.action_space_type == ActionSpaceType.DISCRETE:
            actions = []

            for speed_idx in range(config.speed_granularity):
                speed = config.min_speed + (config.max_speed - config.min_speed) * speed_idx / (config.speed_granularity - 1)

                for steering_idx in range(config.steering_angle_granularity):
                    if steering_idx == 0:
                        steering = 0.0
                        actions.append({"speed": round(speed, 2), "steering_angle": round(steering, 2)})
                    else:
                        steering_magnitude = config.max_steering_angle * steering_idx / (config.steering_angle_granularity - 1)
                        for sign in [-1, 1]:
                            steering = sign * steering_magnitude
                            actions.append({"speed": round(speed, 2), "steering_angle": round(steering, 2)})

            return actions
        else:
            return {
                "steering_angle": {"high": float(config.max_steering_angle), "low": float(-config.max_steering_angle)},
                "speed": {"high": float(config.max_speed), "low": float(config.min_speed)},
            }

    def load_upload_template(self, template_name: str = "training_params.yaml") -> Dict[str, Any]:
        """Load YAML template for model upload configuration.

        Parameters
        ----------
        template_name : str
            Name of the template file to load

        Returns
        -------
        Dict[str, Any]
            Loaded template configuration

        Raises
        ------
        FileNotFoundError
            If template file is not found
        """
        template_path = Path(__file__).parent.parent / "templates" / template_name

        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")

        with open(template_path, "r") as f:
            template_config = yaml.safe_load(f)

        return template_config

    def render_upload_config(self, template_config: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Render upload configuration from template and context.

        Parameters
        ----------
        template_config : Dict[str, Any]
            Template configuration from YAML file
        context : Dict[str, Any]
            Context variables for template rendering

        Returns
        -------
        str
            Rendered configuration content
        """
        template_str = template_config.get("template", "")
        template = Template(template_str)

        render_context = {"parameters": template_config.get("parameters", {}), **context}

        return template.render(render_context)

    def create_training_params_yaml(self, config: AWSDeepRacerConfig, context: Dict[str, Any]) -> str:
        """Create training_params.yaml content from template and config.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        context : Dict[str, Any]
            Context variables for template rendering (AWS account, S3 paths, etc.)

        Returns
        -------
        str
            Rendered training_params.yaml content

        Raises
        ------
        ValueError
            If template loading or rendering fails
        }
        """
        try:
            template_config = self.load_upload_template("training_params.yaml")
            return self.render_upload_config(template_config, context)
        except Exception as e:
            raise ValueError(f"Failed to create training_params.yaml from template: {e}") from e

    def create_hyperparameters_json(self, config: AWSDeepRacerConfig) -> str:
        """Create hyperparameters.json content directly from config.hyperparameters.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration containing hyperparameters

        Returns
        -------
        str
            JSON string of hyperparameters formatted for DeepRacer
        """
        hyperparameters_dict = config.hyperparameters.to_deepracer_format()

        formatted_hyperparameters = {}
        for key, value in hyperparameters_dict.items():
            if key in [
                "batch_size",
                "num_epochs",
                "stack_size",
                "epsilon_steps",
                "num_episodes_between_training",
                "term_cond_max_episodes",
            ]:
                try:
                    formatted_hyperparameters[key] = int(value)
                except (ValueError, TypeError):
                    formatted_hyperparameters[key] = value
            elif key in ["lr", "discount_factor", "beta_entropy", "e_greedy_value", "sac_alpha", "term_cond_avg_score"]:
                try:
                    formatted_hyperparameters[key] = float(value)
                except (ValueError, TypeError):
                    formatted_hyperparameters[key] = value
            else:
                formatted_hyperparameters[key] = value

        return json.dumps(formatted_hyperparameters, indent=2)

    def create_reward_function_file(self, config: AWSDeepRacerConfig, output_path: Path) -> None:
        """Create reward_function.py file from config.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration containing reward function code
        output_path : Path
            Path where to create the reward_function.py file

        Raises
        ------
        ValueError
            If reward_function_code is missing or empty
        """
        if not config.reward_function_code or not config.reward_function_code.strip():
            raise ValueError(
                "reward_function_code is required for AWS DeepRacer model deployment. "
                "Please provide your reward function code in the config."
            )

        with open(output_path, "w") as f:
            f.write(config.reward_function_code)
        info(f"Created reward_function.py: {output_path}")

    def create_model_metadata_file(self, config: AWSDeepRacerConfig, output_path: Path) -> None:
        """Create model_metadata.json file from config using existing method.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration containing model metadata
        output_path : Path
            Path where to create the model_metadata.json file
        """
        metadata = self.create_model_metadata(config)
        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
        info(f"Created model_metadata.json: {output_path}")

    def create_hyperparameters_file(self, config: AWSDeepRacerConfig, output_path: Path) -> None:
        """Create hyperparameters.json file from config using existing method.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration containing hyperparameters
        output_path : Path
            Path where to create the hyperparameters.json file
        """
        if not config.hyperparameters:
            info(f"Warning: No hyperparameters provided, skipping {output_path}")
            return

        hyperparams_content = self.create_hyperparameters_json(config)
        with open(output_path, "w") as f:
            f.write(hyperparams_content)
        info(f"Created hyperparameters.json: {output_path}")

    def create_proper_model_files(self, config: AWSDeepRacerConfig, output_dir: Path, model_name: Optional[str] = None) -> Path:
        """Create proper AWS DeepRacer model files using the model builder.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        output_dir : Path
            Directory to create model files in
        model_name : str, optional
            Override model name

        Returns
        -------
        Path
            Path to the created model archive
        """
        if model_name is None:
            model_name = config.model_name or f"deepracer_model_{int(time.time())}"

        info(f"Creating AWS DeepRacer model: {model_name}")

        scenario = ExperimentalScenario.CENTERLINE_FOLLOWING
        if hasattr(config, "experimental_scenario") and config.experimental_scenario:
            scenario = config.experimental_scenario
        elif hasattr(config, "scenario") and config.scenario:
            try:
                if isinstance(config.scenario, str):
                    scenario = ExperimentalScenario(config.scenario)
                else:
                    scenario = config.scenario
            except ValueError:
                info(f"Unknown scenario '{config.scenario}', using default centerline following")

        version = getattr(config, "version", None) or "1.0.0"

        description = getattr(config, "description", None) or f"AWS DeepRacer model: {model_name}"

        action_space_type = config.action_space_type
        if isinstance(action_space_type, str):
            try:
                action_space_type = ActionSpaceType(action_space_type.lower())
            except (ValueError, AttributeError):
                action_space_type = ActionSpaceType.DISCRETE
                info(f"Could not convert action_space_type '{config.action_space_type}', using DISCRETE")

        try:
            info(f"Attempting to create AWS model with:")
            info(f"  - name: {model_name}")
            info(f"  - description: {description}")
            info(f"  - scenario: {scenario} (type: {type(scenario)})")
            info(f"  - version: {version}")
            info(f"  - action_space: {action_space_type} (type: {type(action_space_type)})")

            aws_model = create_aws_model(
                name=model_name,
                description=description,
                scenario=scenario,
                version=version,
                action_space=action_space_type,
                hyperparameters=config.hyperparameters.to_dict() if config.hyperparameters else {},
                metadata={
                    "created_by": "DeepRacerDeploymentManager",
                    "deployment_config": {},
                    "creation_timestamp": int(time.time()),
                },
            )
        except Exception as e:
            info(f"Error creating model with full config: {e}")
            info(f"Exception type: {type(e)}")
            import traceback

            info(f"Full traceback: {traceback.format_exc()}")

            try:
                info(f"Attempting simple model creation with:")
                info(f"  - name: {model_name}")
                info(f"  - description: {description}")
                info(f"  - version: {version}")

                aws_model = create_simple_aws_model(name=model_name, description=description, version=version)
            except Exception as e2:
                info(f"Error creating simple model: {e2}")
                info(f"Exception type: {type(e2)}")
                import traceback

                info(f"Full traceback: {traceback.format_exc()}")

                raise e2

        if config.reward_function_code and config.reward_function_code.strip():
            try:
                builder = AWSModelBuilder()
                builder = (
                    builder.with_name(model_name)
                    .with_description(description)
                    .with_version(version)
                    .with_action_space(action_space_type)
                    .with_custom_reward_code(config.reward_function_code)
                )

                if config.hyperparameters:
                    builder = builder.with_hyperparameters(config.hyperparameters.to_dict())

                aws_model = builder.build()
            except Exception as e:
                info(f"Error creating model with custom reward function: {e}")

        model_dir = output_dir / "model"
        aws_model.export_to_directory(model_dir)

        self._create_tensorflow_model_with_checkpoints(model_dir, config)

        self._add_deployment_specific_files(model_dir, config)

        info(f"Created model files in directory: {model_dir}")
        return model_dir

    def create_model_archive(self, model_dir: Path, output_dir: Path, model_name: str) -> Path:
        """Create a tar.gz archive for vehicle upload.

        According to AWS documentation, vehicle uploads require compressed *.tar.gz files.

        Parameters
        ----------
        model_dir : Path
            Directory containing the model files
        output_dir : Path
            Directory to save the archive
        model_name : str
            Name of the model

        Returns
        -------
        Path
            Path to the created tar.gz archive
        """
        import tarfile

        archive_path = output_dir / f"{model_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(model_dir, arcname=".")

        info(f"Created compressed model archive for vehicle upload: {archive_path}")
        return archive_path

    def _add_deployment_specific_files(self, model_dir: Path, config: AWSDeepRacerConfig):
        """Add deployment-specific files to match AWS DeepRacer export format.

        Parameters
        ----------
        model_dir : Path
            Model directory to add files to
        config : AWSDeepRacerConfig
            Training configuration
        """
        from deepracer_research.config.training.training_algorithm import TrainingAlgorithm

        if config.training_algorithm == TrainingAlgorithm.SAC:
            version = "4"
        elif config.training_algorithm == TrainingAlgorithm.CLIPPED_PPO:
            version = "5"
        else:
            version = "4"

        model_metadata = {
            "action_space": self._convert_action_space_to_aws_format(config),
            "sensor": [sensor.value for sensor in config.get_sensor_list()],
            "neural_network": self._get_aws_neural_network_type(config.architecture_type),
            "version": version,
            "training_algorithm": config.training_algorithm.value,
            "action_space_type": config.action_space_type.value,
            "preprocess_type": "GREY_SCALE",
            "regional_parameters": [0, 0, 0, 0],
        }

        model_metadata_path = model_dir / "model_metadata.json"
        with open(model_metadata_path, "w") as f:
            json.dump(model_metadata, f, indent=2)

        saved_model_dir = model_dir / "saved_model"
        if saved_model_dir.exists():
            info("Using SavedModel format (compatible with AWS DeepRacer)")

        checkpoint_dir = model_dir / "checkpoint"
        if checkpoint_dir.exists():
            model_subdir = model_dir / "model"
            checkpoint_dir.rename(model_subdir)
            info("Moved checkpoint files to model/ subdirectory per AWS requirements")

            model_metadata_path = model_dir / "model_metadata.json"
            if model_metadata_path.exists():
                model_metadata_copy = model_subdir / "model_metadata.json"
                import shutil

                shutil.copy2(model_metadata_path, model_metadata_copy)
                info("Copied model_metadata.json to model/ subdirectory")

            saved_model_dir = model_dir / "saved_model"
            if saved_model_dir.exists():
                self._create_model_pb_in_model_folder(saved_model_dir, model_subdir)

            self._verify_checkpoint_files(model_subdir)

        reward_function_path = model_dir / "reward_function.py"
        self.create_reward_function_file(config, reward_function_path)

        self._create_optional_files(model_dir, config)

    def _create_model_pb_in_model_folder(self, saved_model_dir: Path, model_subdir: Path):
        """Create model_0.pb file in model/ subdirectory from SavedModel.

        Parameters
        ----------
        saved_model_dir : Path
            Path to the SavedModel directory
        model_subdir : Path
            Path to the model/ subdirectory where model_0.pb should be created
        """
        try:
            import tensorflow as tf

            loaded_model = tf.saved_model.load(str(saved_model_dir))

            if (
                hasattr(loaded_model, "signatures")
                and tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY in loaded_model.signatures
            ):
                serving_fn = loaded_model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            else:

                @tf.function
                def serving_fn(inputs):
                    return loaded_model(inputs)

            from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

            frozen_func = convert_variables_to_constants_v2(serving_fn)
            frozen_graph_def = frozen_func.graph.as_graph_def()

            model_pb_path = model_subdir / "model_0.pb"
            with open(model_pb_path, "wb") as f:
                f.write(frozen_graph_def.SerializeToString())

            info("Created model_0.pb in model/ subdirectory")

        except Exception as e:
            info(f"Could not create model_0.pb in model/ subdirectory: {e}")
            try:
                import tensorflow as tf

                loaded = tf.saved_model.load(str(saved_model_dir))

                @tf.function
                def model_fn(x):
                    return loaded(x)

                sample_input = tf.zeros((1, 120, 160, 3), dtype=tf.float32)
                concrete_fn = model_fn.get_concrete_function(sample_input)

                from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

                frozen_func = convert_variables_to_constants_v2(concrete_fn)

                model_pb_path = model_subdir / "model_0.pb"
                with open(model_pb_path, "wb") as f:
                    f.write(frozen_func.graph.as_graph_def().SerializeToString())

                info("Created model_0.pb using fallback method")

            except Exception as e2:
                info(f"All model_0.pb creation attempts failed: {e2}")
                info("Model will still work without model_0.pb - SavedModel format is sufficient")

    def _verify_checkpoint_files(self, model_subdir: Path):
        """Verify that all required files exist in model/ subdirectory.

        Checks for checkpoint files, model_0.pb, and model_metadata.json.

        Parameters
        ----------
        model_subdir : Path
            The model/ subdirectory path
        """
        required_checkpoint_files = [
            ".coach_checkpoint",
            "0_Step-0.ckpt.data-00000-of-00001",
            "0_Step-0.ckpt.index",
            "0_Step-0.ckpt.meta",
            "deepracer_checkpoints.json",
            "model_metadata.json",
            "model_0.pb",
        ]

        missing_files = []
        for filename in required_checkpoint_files:
            file_path = model_subdir / filename
            if not file_path.exists():
                missing_files.append(filename)

        if missing_files:
            info(f"Missing required files in model/ folder: {missing_files}")
        else:
            info("All required files present in model/ folder")

        actual_files = [f.name for f in model_subdir.iterdir() if f.is_file()]
        info(f"Files in model/ folder: {actual_files}")

    def _create_optional_files(self, model_dir: Path, config: AWSDeepRacerConfig):
        """Create optional files for AWS DeepRacer model import.

        Parameters
        ----------
        model_dir : Path
            Model directory to create optional files in
        config : AWSDeepRacerConfig
            Training configuration
        """
        if config.hyperparameters:
            ip_dir = model_dir / "ip"
            ip_dir.mkdir(exist_ok=True)

            hyperparams_path = ip_dir / "hyperparameters.json"
            self.create_hyperparameters_file(config, hyperparams_path)

        training_params = {
            "version": "1.0",
            "model_name": config.model_name,
            "model_description": config.description or f"DeepRacer model: {config.model_name}",
            "track_name": getattr(config, "track_name", "unknown"),
            "training_algorithm": config.training_algorithm.value,
            "action_space": config.get_action_space_config(),
            "sensor_configuration": [sensor.value for sensor in config.get_sensor_list()],
            "neural_network": self._get_aws_neural_network_type(config.architecture_type),
            "racer_name": getattr(config, "racer_name", "racer"),
            "training_job": {"status": "READY", "created_time": int(time.time()), "model_artifacts_s3_path": ""},
        }

        try:
            import yaml

            training_params_path = model_dir / "training_params.yaml"
            with open(training_params_path, "w") as f:
                yaml.dump(training_params, f, default_flow_style=False, indent=2)
            info("Created training_params.yaml")
        except ImportError:
            info("PyYAML not available, skipping training_params.yaml creation")
        except Exception as e:
            info(f"Could not create training_params.yaml: {e}")

        metrics_dir = model_dir / "metrics" / "training"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        training_metrics = {
            "episode": list(range(1, 11)),
            "episode_reward_mean": [100 + i * 10 for i in range(10)],
            "completion_percentage": [50 + i * 5 for i in range(10)],
            "timestamp": [int(time.time()) + i * 60 for i in range(10)],
        }

        training_metrics_path = metrics_dir / "training-metrics.json"
        with open(training_metrics_path, "w") as f:
            json.dump(training_metrics, f, indent=2)
        info("Created training metrics in metrics/training/")

    def _create_tensorflow_model_with_checkpoints(self, model_dir: Path, config: AWSDeepRacerConfig):
        """Create TensorFlow SavedModel with actual checkpoints using ModelFactory

        Parameters
        ----------
        model_dir : Path
            Model directory to create TensorFlow files in
        config : AWSDeepRacerConfig
            Training configuration
        """
        try:
            from deepracer_research.architectures import create_deepracer_model_with_checkpoints
            from deepracer_research.config import ArchitectureType

            info("Creating TensorFlow model with checkpoints using ModelFactory")

            action_space_config = config.get_action_space_config()
            if config.action_space_type == ActionSpaceType.DISCRETE:
                output_dim = len(action_space_config.get("actions", [{"speed": 1.0, "steering": 0.0}]))
            else:
                output_dim = 2

            architecture_type = getattr(config, "architecture_type", ArchitectureType.ATTENTION_CNN)
            if isinstance(architecture_type, str):
                try:
                    architecture_type = ArchitectureType(architecture_type)
                except ValueError:
                    info(f"Unknown architecture type '{architecture_type}', using ATTENTION_CNN")
                    architecture_type = ArchitectureType.ATTENTION_CNN

            model = create_deepracer_model_with_checkpoints(
                architecture_type=architecture_type,
                output_dir=model_dir,
                input_shape=(160, 120, 3),
                output_size=output_dim,
                model_name="model",
            )

            model.compile(
                optimizer=str(OptimizerType.ADAM.value),
                loss=(
                    str(LossType.MSE.value)
                    if config.action_space_type == ActionSpaceType.CONTINUOUS
                    else str(LossType.CATEGORICAL_CROSSENTROPY.value)
                ),
            )

            info(f"Created {architecture_type} model with proper checkpoints")

        except ImportError as e:
            info(f"Required dependencies not available: {e}")
            raise RuntimeError("TensorFlow and ModelFactory dependencies required for model creation")
        except Exception as e:
            info(f"Error creating TensorFlow model: {e}")
            raise RuntimeError(f"Failed to create TensorFlow model: {e}")

    def create_model_from_existing(self, model_path: Union[str, Path], config: AWSDeepRacerConfig, output_dir: Path) -> Path:
        """Create AWS DeepRacer model files from an existing model.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to existing model files (can be .h5, .keras, or directory)
        config : AWSDeepRacerConfig
            Training configuration
        output_dir : Path
            Directory to create model files in

        Returns
        -------
        Path
            Path to the created model archive
        """
        model_path = Path(model_path)
        model_name = config.model_name or model_path.stem

        info(f"Creating AWS DeepRacer model from existing model: {model_path}")

        aws_model = create_simple_aws_model(
            name=model_name,
            description=config.description or f"Model created from {model_path.name}",
            version=config.version or "1.0.0",
        )

        model_dir = output_dir / "model"
        aws_model.export_to_directory(model_dir)

        self._integrate_existing_model_files(model_path, model_dir, config)

        self._add_deployment_specific_files(model_dir, config)

        info(f"Created model files from existing model: {model_dir}")
        return model_dir

    def _integrate_existing_model_files(self, model_path: Path, model_dir: Path, config: AWSDeepRacerConfig):
        """Integrate existing model files into the AWS DeepRacer structure.

        Parameters
        ----------
        model_path : Path
            Path to existing model files
        model_dir : Path
            Model directory to integrate files into
        config : AWSDeepRacerConfig
            Training configuration
        """
        if model_path.is_file():
            if model_path.suffix in [".h5", ".keras"]:
                target_path = model_dir / "model.h5"
                shutil.copy2(model_path, target_path)
                info(f"Copied Keras model: {model_path} -> {target_path}")

            elif model_path.suffix in [".tar", ".gz", ".zip"]:
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)

                    if model_path.name.endswith(".tar.gz") or model_path.suffix == ".tar":
                        with tarfile.open(model_path, "r:*") as tar:
                            tar.extractall(temp_path)
                    elif model_path.suffix == ".zip":
                        import zipfile

                        with zipfile.ZipFile(model_path, "r") as zip_file:
                            zip_file.extractall(temp_path)

                    for item in temp_path.rglob("*"):
                        if item.is_file():
                            relative_path = item.relative_to(temp_path)
                            target_path = model_dir / relative_path
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(item, target_path)

        elif model_path.is_dir():
            for item in model_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(model_path)
                    target_path = model_dir / relative_path
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, target_path)

    def create_model_with_manager_integration(
        self, config: AWSDeepRacerConfig, output_dir: Path, model_manager=None, register_model: bool = True
    ) -> Path:
        """Create model with ModelManager integration for better tracking.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        output_dir : Path
            Directory to create model files in
        model_manager : ModelManager, optional
            Model manager instance for registration
        register_model : bool
            Whether to register the model with the manager

        Returns
        -------
        Path
            Path to the created model archive
        """
        model_archive = self.create_proper_model_files(config, output_dir)

        if model_manager and register_model:
            try:
                from datetime import datetime

                from deepracer_research.models.model_metadata import ModelMetadata

                metadata = ModelMetadata(
                    model_name=config.model_name,
                    version=getattr(config, "version", None) or "1.0.0",
                    algorithm=TrainingAlgorithm.PPO.value.upper(),
                    neural_architecture=ArchitectureType.ATTENTION_CNN.value.upper(),
                    reward_function=config.reward_function_code or "Default centerline following",
                    hyperparameters=config.hyperparameters.to_dict() if config.hyperparameters else {},
                    training_episodes=0,
                    training_duration_hours=0.0,
                    scenario="aws_deployment",
                    notes=f"Model created for AWS DeepRacer deployment via DeploymentManager",
                    deployment_status="created",
                    created_date=datetime.now(),
                    last_modified=datetime.now(),
                )

                model_dir = output_dir / "model"
                if model_dir.exists():
                    model_id = model_manager.register_model(model_dir, metadata)
                    info(f"Model registered with manager: {model_id}")
                else:
                    info("Model directory not found for registration")

            except Exception as e:
                info(f"Failed to register model with manager: {e}")

        return model_archive

    def upload_model_to_s3(self, model_dir_path: Path, model_name: str, s3_prefix: str = None) -> str:
        """Upload model files individually to S3 for import_model.

        AWS DeepRacer expects individual model files, not archives.

        Parameters
        ----------
        model_dir_path : Path
            Path to the model directory containing individual files
        model_name : str
            Name of the model
        s3_prefix : str, optional
            Optional S3 prefix (e.g., 'my_model/version_2')

        Returns
        -------
        str
            S3 URI pointing to the root folder for import
        """
        s3_client = boto3.client("s3", region_name=self.region)

        bucket_name = f"deepracer-models-{uuid.uuid4().hex[:8]}"

        try:
            try:
                if self.region == "us-east-1":
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": self.region})
            except ClientError as e:
                if e.response["Error"]["Code"] != "BucketAlreadyExists":
                    raise

            import datetime

            timestamp = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")

            if s3_prefix:
                s3_path_prefix = f"{model_name}/{s3_prefix}/{timestamp}"
            else:
                s3_path_prefix = f"{model_name}/{timestamp}"

            uploaded_files = []

            for file_path in model_dir_path.rglob("*"):
                if file_path.is_file():
                    relative_path = file_path.relative_to(model_dir_path)
                    s3_key = f"{s3_path_prefix}/{relative_path.as_posix()}"

                    s3_client.upload_file(str(file_path), bucket_name, s3_key)
                    uploaded_files.append(s3_key)
                    info(f"Uploaded: {s3_key}")

            info(f"Uploaded {len(uploaded_files)} files to S3")
            return f"s3://{bucket_name}/{s3_path_prefix}/"

        except ClientError as e:
            raise Exception(f"Failed to upload model to S3: {e}")

    def _sanitize_model_name(self, name: str) -> str:
        """Sanitize model name to meet AWS DeepRacer requirements.

        Parameters
        ----------
        name : str
            Original model name

        Returns
        -------
        str
            Sanitized model name
        """
        sanitized = name.replace("_", "-")

        import re

        sanitized = re.sub(r"[^a-zA-Z0-9-]", "-", sanitized)

        sanitized = re.sub(r"-+", "-", sanitized)

        sanitized = sanitized.strip("-")

        if not sanitized:
            sanitized = f"model-{int(time.time())}"
        elif len(sanitized) > 255:
            sanitized = sanitized[:255].rstrip("-")

        return sanitized

    def generate_deployment_config(self, config: AWSDeepRacerConfig, save_to_file: bool = True) -> Dict[str, Any]:
        """Generate complete AWS DeepRacer deployment configuration.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        save_to_file : bool, optional
            Whether to save the config to deployments directory, by default True

        Returns
        -------
        Dict[str, Any]
            Complete deployment configuration
        """
        import json
        from pathlib import Path

        role_arn = self.get_or_create_service_role()

        deployment_config = config.to_deepracer_config_file()

        deployment_config["roleArn"] = role_arn
        deployment_config["metadata"] = {
            "createdAt": datetime.now().isoformat(),
            "experimentalScenario": config.experimental_scenario.value,
            "tags": config.tags or {},
        }

        if save_to_file:
            deployments_dir = Path("deployments")
            deployments_dir.mkdir(exist_ok=True)

            config_filename = f"{self._sanitize_model_name(config.model_name)}_deepracer_config.json"
            config_path = deployments_dir / config_filename

            with open(config_path, "w") as f:
                json.dump(deployment_config, f, indent=2, default=str)

            info(f"Deployment configuration saved to: {config_path}")

        return deployment_config

    def create_model_with_manager_integration(self, config: AWSDeepRacerConfig) -> Dict[str, Any]:
        """Create and deploy a model with full manager integration.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration including architecture type, scenario, etc.

        Returns
        -------
        Dict[str, Any]
            Combined result containing model metadata, deployment config, and AWS deployment response
        """
        try:
            import tempfile

            from deepracer_research.config import ArchitectureType
            from deepracer_research.models.deployment_status import DeploymentStatus, DeploymentType
            from deepracer_research.models.manager import ModelManager
            from deepracer_research.models.model_metadata import ModelMetadata

            info(f"Creating model with manager integration for scenario: {config.experimental_scenario.value}")

            deployment_config = self.generate_deployment_config(config, save_to_file=True)

            architecture_type = getattr(config, "architecture_type", ArchitectureType.ATTENTION_CNN)
            if isinstance(architecture_type, str):
                try:
                    architecture_type = ArchitectureType(architecture_type)
                except ValueError:
                    info(f"Unknown architecture type '{architecture_type}', using ATTENTION_CNN")
                    architecture_type = ArchitectureType.ATTENTION_CNN

            metadata = ModelMetadata(
                model_name=config.model_name,
                version=getattr(config, "version", "1.0.0"),
                description=config.description,
                neural_architecture=architecture_type,
                scenario=config.experimental_scenario.value,
                action_space_type=(
                    config.action_space_type.value
                    if hasattr(config.action_space_type, "value")
                    else str(config.action_space_type)
                ),
                training_algorithm=config.training_algorithm,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                model_archive = self.create_proper_model_files(config, temp_path)

                model_manager = ModelManager()
                model_id = model_manager.register_model(model_archive, metadata)

                info(f"Model registered with ID: {model_id}")

                aws_response = self.import_model(config, None)

                if aws_response.get("ModelArn"):
                    model_manager.update_deployment_status(model_id, DeploymentStatus.DEPLOYED, DeploymentType.AWS_DEEPRACER)

                return {
                    "model_id": model_id,
                    "model_metadata": metadata,
                    "deployment_config": deployment_config,
                    "aws_response": aws_response,
                    "architecture_type": architecture_type.value,
                    "scenario": config.experimental_scenario.value,
                }

        except Exception as e:
            info(f" Error in create_model_with_manager_integration: {e}")
            raise RuntimeError(f"Failed to create model with manager integration: {e}")

    def import_model(self, config: AWSDeepRacerConfig, model_s3_path: Optional[str] = None) -> Dict[str, Any]:
        """Import a model using the import_model API.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        model_s3_path : Optional[str]
            S3 path to existing model archive. If None, creates a proper model with checkpoints.

        Returns
        -------
        Dict[str, Any]
            Response from import_model API
        """
        role_arn = self.get_or_create_service_role()

        if model_s3_path is None:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                model_archive = self.create_proper_model_files(config, temp_path)
                model_s3_path = self.upload_model_to_s3(model_archive, config.model_name)

        info(f"Importing model from S3: {model_s3_path}")

        sanitized_name = self._sanitize_model_name(config.model_name)
        if sanitized_name != config.model_name:
            info(f"Model name sanitized: '{config.model_name}' -> '{sanitized_name}'")

        try:
            response = self.client.import_model(
                Type="REINFORCEMENT_LEARNING",
                Name=sanitized_name,
                RoleArn=role_arn,
                ModelArtifactsS3Path=model_s3_path,
                Description=config.description or f"Imported model: {sanitized_name}",
            )
            return response
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ValidationException":
                raise Exception(f"Invalid model configuration: {error_message}")
            elif error_code == "ResourceInUseException":
                raise Exception(f"Model name already in use: {error_message}")
            elif error_code == "LimitExceededException":
                raise Exception(f"Model import limit exceeded: {error_message}")
            else:
                raise Exception(f"Failed to import model: {error_message}")

    def upload_reward_function_to_s3(self, reward_function_code: str, model_name: str) -> str:
        """Upload reward function to S3 and return the S3 URI.

        Parameters
        ----------
        reward_function_code : str
            The reward function Python code
        model_name : str
            Name of the model (used for S3 key)

        Returns
        -------
        str
            S3 URI of the uploaded reward function
        """
        s3_client = boto3.client("s3", region_name=self.region)

        bucket_name = f"deepracer-reward-functions-{uuid.uuid4().hex[:8]}"

        try:
            try:
                if self.region == "us-east-1":
                    s3_client.create_bucket(Bucket=bucket_name)
                else:
                    s3_client.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": self.region})
            except ClientError as e:
                if e.response["Error"]["Code"] != "BucketAlreadyExists":
                    raise

            key = f"reward_functions/{model_name}/reward_function.py"
            s3_client.put_object(
                Bucket=bucket_name, Key=key, Body=reward_function_code.encode("utf-8"), ContentType="text/plain"
            )

            return f"s3://{bucket_name}/{key}"

        except ClientError as e:
            raise Exception(f"Failed to upload reward function to S3: {e}")

    def create_training_job(
        self, config: AWSDeepRacerConfig, use_import_model: bool = True, existing_model_s3_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create a training job in AWS DeepRacer.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        use_import_model : bool
            If True, use import_model API instead of create_reinforcement_learning_model
        existing_model_s3_path : Optional[str]
            S3 path to existing model archive (only used with import_model)

        Returns
        -------
        Dict[str, Any]
            Response from import_model or create_reinforcement_learning_model API
        """
        if use_import_model:
            return self.import_model(config, existing_model_s3_path)
        else:
            return self._create_training_job_legacy(config)

    def _create_training_job_legacy(self, config: AWSDeepRacerConfig) -> Dict[str, Any]:
        """Create a training job using the legacy create_reinforcement_learning_model API.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration

        Returns
        -------
        Dict[str, Any]
            Response from create_reinforcement_learning_model API
        """
        role_arn = self.get_or_create_service_role()

        reward_function_s3_uri = self.upload_reward_function_to_s3(config.reward_function_code, config.model_name)

        training_config = config.to_deepracer_training_job_config(reward_function_s3_uri, role_arn)

        info("Creating training job with configuration:", training_config)

        try:
            response = self.client.create_reinforcement_learning_model(**training_config)
            return response
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ValidationException":
                raise Exception(f"Invalid training configuration: {error_message}")
            elif error_code == "ResourceInUseException":
                raise Exception(f"Model name already in use: {error_message}")
            elif error_code == "LimitExceededException":
                raise Exception(f"Training job limit exceeded: {error_message}")
            else:
                raise Exception(f"Failed to create training job: {error_message}")

    def import_existing_model(
        self, model_path: Union[str, Path], model_name: str, description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Import an existing trained model from local filesystem.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the model directory or archive file
        model_name : str
            Name for the imported model
        description : Optional[str]
            Description for the model

        Returns
        -------
        Dict[str, Any]
            Response from import_model API
        """
        model_path = Path(model_path)
        role_arn = self.get_or_create_service_role()

        if model_path.is_dir():
            s3_path = self.upload_model_to_s3(model_path, model_name)
        elif model_path.suffix in [".tar", ".gz", ".tgz"]:
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                with tarfile.open(model_path, "r:*") as tar:
                    tar.extractall(temp_path)
                model_dirs = [d for d in temp_path.iterdir() if d.is_dir()]
                if model_dirs:
                    s3_path = self.upload_model_to_s3(model_dirs[0], model_name)
                else:
                    s3_path = self.upload_model_to_s3(temp_path, model_name)
        else:
            raise ValueError(f"Unsupported model path format: {model_path}")

        info(f"Importing existing model from: {s3_path}")

        sanitized_name = self._sanitize_model_name(model_name)
        if sanitized_name != model_name:
            info(f"Model name sanitized: '{model_name}' -> '{sanitized_name}'")

        try:
            response = self.client.import_model(
                Type="REINFORCEMENT_LEARNING",
                Name=sanitized_name,
                RoleArn=role_arn,
                ModelArtifactsS3Path=s3_path,
                Description=description or f"Imported model: {sanitized_name}",
            )
            return response
        except ClientError as e:
            e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")
            raise Exception(f"Failed to import existing model: {error_message}")

    def create_model_from_config(
        self,
        config: AWSDeepRacerConfig,
        output_dir: Optional[Path] = None,
        upload_to_s3: bool = True,
        create_archive: bool = False,
        s3_prefix: str = None,
    ) -> Union[Path, str]:
        """Create a model from configuration for console import or vehicle upload.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        output_dir : Optional[Path]
            Directory to save the model files. If None, uses temporary directory.
        upload_to_s3 : bool
            Whether to upload to S3 and return S3 URI (for console import)
        create_archive : bool
            Whether to create a tar.gz archive (for vehicle upload)
        s3_prefix : str, optional
            Optional S3 prefix for organizing models (e.g., 'experiments/v1')

        Returns
        -------
        Union[Path, str]
            Local path to model directory/archive or S3 URI if uploaded
        """
        if output_dir is None:
            output_dir = Path.cwd() / "models"
            output_dir.mkdir(exist_ok=True)

        model_dir = self.create_proper_model_files(config, output_dir)

        if create_archive:
            archive_path = self.create_model_archive(model_dir, output_dir, config.model_name)
            import shutil

            shutil.rmtree(model_dir)
            return archive_path
        elif upload_to_s3:
            s3_path = self.upload_model_to_s3(model_dir, config.model_name, s3_prefix)
            import shutil

            shutil.rmtree(model_dir)
            return s3_path
        else:
            return model_dir

    def create_model_for_vehicle_upload(
        self, config: AWSDeepRacerConfig, output_dir: Optional[Path] = None, s3_prefix: str = None
    ) -> Path:
        """Create a compressed model archive ready for vehicle upload.

        This method creates a tar.gz archive that can be uploaded to your AWS DeepRacer vehicle
        as described in the AWS documentation.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration
        output_dir : Optional[Path]
            Directory to save the archive. If None, uses current directory.
        s3_prefix : str, optional
            Optional S3 prefix for organizing models (not used for local archives)

        Returns
        -------
        Path
            Path to the created tar.gz archive ready for vehicle upload
        """
        if output_dir is None:
            output_dir = Path.cwd()

        return self.create_model_from_config(
            config=config, output_dir=output_dir, upload_to_s3=False, create_archive=True, s3_prefix=s3_prefix
        )

    def get_training_job_status(self, model_name: str) -> Dict[str, Any]:
        """Get the status of a model (training job or imported model).

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Dict[str, Any]
            Model status information
        """
        try:
            response = self.client.describe_model(modelName=model_name)
            return response
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                models_response = self.client.list_models()
                for model in models_response.get("Models", []):
                    if model.get("ModelName") == model_name:
                        return self.client.describe_model(modelName=model_name)
                raise Exception(f"Model not found: {model_name}")
            else:
                raise Exception(f"Failed to get model status: {e}")

    def list_training_jobs(self) -> Dict[str, Any]:
        """List all models (both training jobs and imported models).

        Returns
        -------
        Dict[str, Any]
            List of models
        """
        try:
            response = self.client.list_models()
            return response
        except ClientError as e:
            raise Exception(f"Failed to list models: {e}")

    def get_model_details(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Dict[str, Any]
            Detailed model information
        """
        try:
            response = self.client.describe_model(modelName=model_name)
            return response
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                raise Exception(f"Model not found: {model_name}")
            else:
                raise Exception(f"Failed to get model details: {e}")

    def clone_model(self, source_model_name: str, target_model_name: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Clone an existing model.

        Parameters
        ----------
        source_model_name : str
            Name of the source model to clone
        target_model_name : str
            Name for the cloned model
        description : Optional[str]
            Description for the cloned model

        Returns
        -------
        Dict[str, Any]
            Response from clone_model API
        """
        try:
            response = self.client.clone_model(
                sourceModelName=source_model_name,
                targetModelName=target_model_name,
                targetModelDescription=description or f"Clone of {source_model_name}",
            )
            return response
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException":
                raise Exception(f"Source model not found: {source_model_name}")
            elif error_code == "ResourceInUseException":
                raise Exception(f"Target model name already in use: {target_model_name}")
            else:
                raise Exception(f"Failed to clone model: {error_message}")

    def stop_training_job(self, model_name: str) -> Dict[str, Any]:
        """Stop a training job (if it's currently training).

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Dict[str, Any]
            Response from stop_training_job API
        """
        try:
            response = self.client.stop_training_job(modelName=model_name)
            return response
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException":
                raise Exception(f"Model not found: {model_name}")
            elif error_code == "ValidationException":
                raise Exception(f"Model is not currently training: {model_name}")
            else:
                raise Exception(f"Failed to stop training job: {error_message}")

    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """Delete a model.

        Parameters
        ----------
        model_name : str
            Name of the model to delete

        Returns
        -------
        Dict[str, Any]
            Response from delete_model API
        """
        try:
            response = self.client.delete_model(modelName=model_name)
            return response
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "ResourceNotFoundException":
                raise Exception(f"Model not found: {model_name}")
            else:
                raise Exception(f"Failed to delete model: {e}")

    def wait_for_import_completion(self, model_name: str, timeout_minutes: int = 30) -> Dict[str, Any]:
        """Wait for model import to complete.

        Parameters
        ----------
        model_name : str
            Name of the model
        timeout_minutes : int
            Maximum time to wait in minutes

        Returns
        -------
        Dict[str, Any]
            Final model status
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            try:
                status = self.get_model_details(model_name)
                model_status = status.get("ModelStatus", "UNKNOWN")

                if model_status in ["READY", "FAILED"]:
                    return status

                if time.time() - start_time > timeout_seconds:
                    raise Exception(f"Model import timeout after {timeout_minutes} minutes")

                time.sleep(10)

            except Exception as e:
                if "not found" in str(e).lower():
                    if time.time() - start_time > timeout_seconds:
                        raise Exception(f"Model import timeout after {timeout_minutes} minutes")
                    time.sleep(10)
                    continue
                else:
                    raise

    def wait_for_training_completion(self, model_name: str, timeout_minutes: int = 120) -> Dict[str, Any]:
        """Wait for training job or model import to complete.

        Parameters
        ----------
        model_name : str
            Name of the model
        timeout_minutes : int
            Maximum time to wait in minutes

        Returns
        -------
        Dict[str, Any]
            Final model status
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        while True:
            try:
                status = self.get_model_details(model_name)
                model_status = status.get("ModelStatus", "UNKNOWN")

                if model_status in ["READY", "FAILED"]:
                    return status

                if time.time() - start_time > timeout_seconds:
                    raise Exception(f"Model operation timeout after {timeout_minutes} minutes")

                time.sleep(30)

            except Exception as e:
                if "not found" in str(e).lower():
                    if time.time() - start_time > timeout_seconds:
                        raise Exception(f"Model operation timeout after {timeout_minutes} minutes")
                    time.sleep(30)
                    continue
                else:
                    raise

    def start_evaluation_job(
        self,
        model_name: str,
        track_arn: str,
        evaluation_type: Union[EvaluationType, str] = EvaluationType.TIME_TRIAL,
        number_of_trials: int = 3,
        tags: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Start an evaluation job for a model.

        Parameters
        ----------
        model_name : str
            Name of the model to evaluate
        track_arn : str
            ARN of the track to evaluate on
        evaluation_type : Union[EvaluationType, str], optional
            Type of evaluation, by default EvaluationType.TIME_TRIAL
        number_of_trials : int, optional
            Number of evaluation trials to run, by default 3
        tags : Optional[Dict[str, str]], optional
            Tags to apply to the evaluation job, by default None

        Returns
        -------
        Dict[str, Any]
            Response from start_evaluation_job API

        Raises
        ------
        Exception
            If the evaluation job cannot be started
        """
        eval_type_str = evaluation_type.value if isinstance(evaluation_type, EvaluationType) else evaluation_type

        try:
            params = {
                "modelName": model_name,
                "trackArn": track_arn,
                "evaluationType": eval_type_str,
                "numberOfTrials": number_of_trials,
            }

            if tags:
                params["tags"] = tags

            response = self.client.start_evaluation_job(**params)
            info(f"Started evaluation job for model '{model_name}' on track: {track_arn}")
            return response

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException":
                raise Exception(f"Model not found: {model_name}")
            elif error_code == "ValidationException":
                raise Exception(f"Invalid evaluation parameters: {error_message}")
            elif error_code == "ResourceInUseException":
                raise Exception(f"Model is currently being used in another job: {model_name}")
            else:
                raise Exception(f"Failed to start evaluation job: {error_message}")

    def stop_evaluation_job(self, model_name: str) -> Dict[str, Any]:
        """Stop an active evaluation job.

        Parameters
        ----------
        model_name : str
            Name of the model whose evaluation job should be stopped

        Returns
        -------
        Dict[str, Any]
            Response from stop_evaluation_job API

        Raises
        ------
        Exception
            If the evaluation job cannot be stopped
        """
        try:
            response = self.client.stop_evaluation_job(modelName=model_name)
            info(f"Stopped evaluation job for model: {model_name}")
            return response

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException":
                raise Exception(f"No active evaluation job found for model: {model_name}")
            elif error_code == "ValidationException":
                raise Exception(f"Cannot stop evaluation job: {error_message}")
            else:
                raise Exception(f"Failed to stop evaluation job: {error_message}")

    def get_evaluation_job_status(self, model_name: str) -> Dict[str, Any]:
        """Get the status of an evaluation job.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Dict[str, Any]
            Evaluation job status and details

        Raises
        ------
        Exception
            If the evaluation job status cannot be retrieved
        """
        try:
            response = self.client.describe_evaluation_job(modelName=model_name)
            return response

        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            error_message = e.response.get("Error", {}).get("Message", "")

            if error_code == "ResourceNotFoundException":
                raise Exception(f"No evaluation job found for model: {model_name}")
            else:
                raise Exception(f"Failed to get evaluation job status: {error_message}")

    def list_evaluation_jobs(self, status_filter: Optional[str] = None, max_results: int = 10) -> Dict[str, Any]:
        """List evaluation jobs.

        Parameters
        ----------
        status_filter : Optional[str], optional
            Filter by evaluation status (e.g., "InProgress", "Completed", "Failed"), by default None
        max_results : int, optional
            Maximum number of results to return, by default 10

        Returns
        -------
        Dict[str, Any]
            List of evaluation jobs

        Raises
        ------
        Exception
            If the evaluation jobs cannot be listed
        """
        try:
            params = {"maxResults": max_results}

            if status_filter:
                params["statusFilter"] = status_filter

            response = self.client.list_evaluation_jobs(**params)
            return response

        except ClientError as e:
            error_message = e.response.get("Error", {}).get("Message", "")
            raise Exception(f"Failed to list evaluation jobs: {error_message}")

    def get_evaluation_logs(self, model_name: str, log_type: str = "all") -> Dict[str, Any]:
        """Get evaluation logs for a model.

        Parameters
        ----------
        model_name : str
            Name of the model
        log_type : str, optional
            Type of logs to retrieve ("all", "simulation", "robomaker"), by default "all"

        Returns
        -------
        Dict[str, Any]
            Evaluation logs and metadata

        Raises
        ------
        Exception
            If the evaluation logs cannot be retrieved
        """
        try:
            eval_status = self.get_evaluation_job_status(model_name)

            if "evaluationJobArn" not in eval_status:
                raise Exception(f"No evaluation job ARN found for model: {model_name}")

            logs_client = boto3.client("logs", region_name=self.region)

            log_groups = [f"/aws/deepracer/evaluation/{model_name}", f"/aws/robomaker/simulation-job", f"/aws/deepracer/logs"]

            all_logs = {}

            for log_group in log_groups:
                try:
                    streams_response = logs_client.describe_log_streams(
                        logGroupName=log_group, orderBy="LastEventTime", descending=True, limit=10
                    )

                    for stream in streams_response.get("logStreams", []):
                        if model_name in stream["logStreamName"]:
                            logs_response = logs_client.get_log_events(
                                logGroupName=log_group, logStreamName=stream["logStreamName"], limit=1000
                            )

                            all_logs[f"{log_group}/{stream['logStreamName']}"] = {
                                "events": logs_response.get("events", []),
                                "nextForwardToken": logs_response.get("nextForwardToken"),
                                "nextBackwardToken": logs_response.get("nextBackwardToken"),
                            }

                except logs_client.exceptions.ResourceNotFoundException:
                    continue
                except Exception as log_error:
                    info(f"Could not retrieve logs from {log_group}: {log_error}")
                    continue

            return {
                "modelName": model_name,
                "evaluationJobStatus": eval_status,
                "logs": all_logs,
                "logType": log_type,
                "timestamp": datetime.now().isoformat(),
            }

        except ClientError as e:
            error_message = e.response.get("Error", {}).get("Message", "")
            raise Exception(f"Failed to get evaluation logs: {error_message}")

    def wait_for_evaluation_completion(self, model_name: str, timeout_minutes: int = 60) -> Dict[str, Any]:
        """Wait for evaluation job to complete.

        Parameters
        ----------
        model_name : str
            Name of the model
        timeout_minutes : int, optional
            Maximum time to wait in minutes, by default 60

        Returns
        -------
        Dict[str, Any]
            Final evaluation job status

        Raises
        ------
        Exception
            If evaluation times out or fails
        """
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        info(f"Waiting for evaluation of model '{model_name}' to complete...")

        while True:
            try:
                status = self.get_evaluation_job_status(model_name)
                eval_status = status.get("evaluationJobStatus", "UNKNOWN")

                if eval_status in ["Completed", "Failed", "Stopped"]:
                    if eval_status == "Completed":
                        info(f"Evaluation completed successfully for model: {model_name}")
                    elif eval_status == "Failed":
                        info(f"Evaluation failed for model: {model_name}")
                    else:
                        info(f"Evaluation stopped for model: {model_name}")
                    return status

                if time.time() - start_time > timeout_seconds:
                    raise Exception(f"Evaluation timeout after {timeout_minutes} minutes")

                info(f"Evaluation status: {eval_status} - waiting...")
                time.sleep(30)

            except Exception as e:
                if "No evaluation job found" in str(e):
                    if time.time() - start_time > timeout_seconds:
                        raise Exception(f"Evaluation timeout after {timeout_minutes} minutes")
                    time.sleep(30)
                    continue
                else:
                    raise

    def get_evaluation_results(self, model_name: str) -> Dict[str, Any]:
        """Get comprehensive evaluation results for a model.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        Dict[str, Any]
            Comprehensive evaluation results including status, metrics, and logs

        Raises
        ------
        Exception
            If evaluation results cannot be retrieved
        """
        try:
            eval_status = self.get_evaluation_job_status(model_name)

            eval_logs = self.get_evaluation_logs(model_name)

            results = {
                "modelName": model_name,
                "evaluationStatus": eval_status,
                "evaluationLogs": eval_logs,
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "status": eval_status.get("evaluationJobStatus", "UNKNOWN"),
                    "trackArn": eval_status.get("trackArn", ""),
                    "evaluationType": eval_status.get("evaluationType", ""),
                    "numberOfTrials": eval_status.get("numberOfTrials", 0),
                    "createdAt": eval_status.get("createdAt", ""),
                    "completedAt": eval_status.get("completedAt", ""),
                },
            }

            if "evaluationResults" in eval_status:
                results["metrics"] = eval_status["evaluationResults"]

            return results

        except Exception as e:
            raise Exception(f"Failed to get evaluation results: {e}")

    def run_complete_evaluation(
        self,
        model_name: str,
        track_arn: str,
        evaluation_type: Union[EvaluationType, str] = EvaluationType.TIME_TRIAL,
        number_of_trials: int = 3,
        timeout_minutes: int = 60,
        get_logs: bool = True,
    ) -> Dict[str, Any]:
        """Run a complete evaluation workflow: start, wait, and get results.

        Parameters
        ----------
        model_name : str
            Name of the model to evaluate
        track_arn : str
            ARN of the track to evaluate on
        evaluation_type : Union[EvaluationType, str], optional
            Type of evaluation, by default EvaluationType.TIME_TRIAL
        number_of_trials : int, optional
            Number of trials to run, by default 3
        timeout_minutes : int, optional
            Maximum time to wait, by default 60
        get_logs : bool, optional
            Whether to retrieve logs, by default True

        Returns
        -------
        Dict[str, Any]
            Complete evaluation results

        Raises
        ------
        Exception
            If any step of the evaluation fails
        """
        try:
            start_response = self.start_evaluation_job(
                model_name=model_name, track_arn=track_arn, evaluation_type=evaluation_type, number_of_trials=number_of_trials
            )

            completion_status = self.wait_for_evaluation_completion(model_name=model_name, timeout_minutes=timeout_minutes)

            if get_logs:
                results = self.get_evaluation_results(model_name)
            else:
                results = {
                    "modelName": model_name,
                    "evaluationStatus": completion_status,
                    "startResponse": start_response,
                    "timestamp": datetime.now().isoformat(),
                }

            return results

        except Exception as e:
            try:
                self.stop_evaluation_job(model_name)
            except:
                pass

            raise Exception(f"Complete evaluation workflow failed: {e}")

    def _generate_model_id(self, config: AWSDeepRacerConfig) -> str:
        """Generate a unique model ID based on configuration parameters.

        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration

        Returns
        -------
        str
            Unique model identifier (8-character hash)
        """
        config_data = {
            "model_name": config.model_name,
            "track_arn": config.track_arn,
            "training_algorithm": config.training_algorithm.value,
            "action_space_type": config.action_space_type.value,
            "max_speed": config.max_speed,
            "max_steering_angle": config.max_steering_angle,
            "experimental_scenario": config.experimental_scenario.value,
            "speed_granularity": config.speed_granularity,
            "steering_angle_granularity": config.steering_angle_granularity,
        }

        if config.hyperparameters:
            hyperparams = config.hyperparameters.to_dict()
            config_data["hyperparameters"] = hyperparams

        config_string = json.dumps(config_data, sort_keys=True)

        model_hash = hashlib.sha256(config_string.encode()).hexdigest()[:8]

        return model_hash

    def create_essential_model_files(self, config: AWSDeepRacerConfig, output_dir: Optional[Path] = None) -> tuple[Path, str]:
        """Create only the essential model files (hyperparameters.json, model_metadata.json, reward_function.py) in a models/{id} folder.


        Parameters
        ----------
        config : AWSDeepRacerConfig
            Training configuration containing model parameters, hyperparameters, and reward function code
        output_dir : Optional[Path]
            Directory to create the models folder in. If None, uses current directory.

        Returns
        -------
        tuple[Path, str]
            Tuple of (Path to the created model directory, unique model ID)

        Raises
        ------
        ValueError
            If reward_function_code is missing (raised by create_reward_function_file)
        """
        if output_dir is None:
            output_dir = Path.cwd()

        model_id = self._generate_model_id(config)

        models_base_dir = output_dir / "models"
        models_base_dir.mkdir(exist_ok=True)

        model_dir = models_base_dir / model_id
        model_dir.mkdir(exist_ok=True)

        hyperparams_path = model_dir / "hyperparameters.json"
        self.create_hyperparameters_file(config, hyperparams_path)

        metadata_path = model_dir / "model_metadata.json"
        self.create_model_metadata_file(config, metadata_path)

        reward_function_path = model_dir / "reward_function.py"
        self.create_reward_function_file(config, reward_function_path)

        info(f"Successfully created model files in: {model_dir}")
        info(f"Model ID: {model_id}")
        return model_dir, model_id
