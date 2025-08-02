import hashlib
import json
import shutil
import tarfile
import tempfile
import traceback
import zipfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import boto3

from deepracer_research.deployment import AWSDeepRacerConfig
from deepracer_research.models.model_metadata import ModelMetadata
from deepracer_research.models.model_version import ModelVersion
from deepracer_research.utils import error, get_deepracer_client, get_s3_client, info


class ModelManager:
    """Comprehensive model lifecycle management for AWS DeepRacer research.

    Attributes
    ----------
    config : Any
        Configuration object with storage paths and settings
    model_registry : Dict[str, ModelMetadata]
        In-memory model metadata registry
    version_history : Dict[str, List[ModelVersion]]
        Model version tracking
    aws_session : boto3.Session
        AWS session for cloud operations
    s3_client : boto3.client
        AWS S3 client
    deepracer_client : boto3.client
        AWS DeepRacer client
    sagemaker_client : boto3.client
        AWS SageMaker client
    """

    def __init__(self, config: Optional[Any] = None, aws_session: Optional[boto3.Session] = None) -> None:
        """Initialize the model manager with configuration and AWS session.

        Parameters
        ----------
        config : Any, optional
            Configuration object with storage paths and AWS settings
        aws_session : boto3.Session, optional
            Pre-configured AWS session
        """
        from deepracer_research.config import ResearchConstants

        self.config = config or ResearchConstants()
        self.model_registry: Dict[str, ModelMetadata] = {}
        self.version_history: Dict[str, List[ModelVersion]] = {}

        self._setup_directories()

        self.aws_session = aws_session
        self.s3_client = None
        self.deepracer_client = None
        self.sagemaker_client = None

        if self.aws_session is None:
            try:
                self.aws_session = boto3.Session(region_name=self.config.AWS_REGION)

                self.s3_client = get_s3_client(region_name=self.config.AWS_REGION, aws_session=self.aws_session)
                self.deepracer_client = get_deepracer_client(region_name=self.config.AWS_REGION, aws_session=self.aws_session)
                self.sagemaker_client = self.aws_session.client("sagemaker")
                info("AWS clients initialized successfully")
            except Exception as e:
                error(f"AWS initialization failed", {"error": str(e), "error_details": traceback.format_exc()})
                self.aws_session = None

        self._load_registry()

    def _setup_directories(self) -> None:
        """Create necessary storage directories."""
        directories = [self.config.MODEL_STORAGE_PATH, self.config.EXPORT_STORAGE_PATH, self.config.IMPORT_STORAGE_PATH]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

        info("Model storage directories initialized")

    def _load_registry(self) -> None:
        """Load existing model registry from storage."""
        registry_file = Path(self.config.MODEL_STORAGE_PATH) / "model_registry.json"

        if registry_file.exists():
            try:
                with open(registry_file, "r") as f:
                    registry_data = json.load(f)

                for model_id, metadata_dict in registry_data.items():
                    metadata_dict["created_date"] = datetime.fromisoformat(metadata_dict["created_date"])
                    metadata_dict["last_modified"] = datetime.fromisoformat(metadata_dict["last_modified"])
                    self.model_registry[model_id] = ModelMetadata(**metadata_dict)

                info(f"Loaded {len(self.model_registry)} models from registry")

            except Exception as e:
                error(f"Failed to load model registry", {"error": str(e), "error_details": traceback.format_exc()})
                self.model_registry = {}

    def _save_registry(self) -> None:
        """Save model registry to storage."""
        registry_file = Path(self.config.MODEL_STORAGE_PATH) / "model_registry.json"

        try:
            registry_data = {}
            for model_id, metadata in self.model_registry.items():
                metadata_dict = asdict(metadata)
                metadata_dict["created_date"] = metadata.created_date.isoformat()
                metadata_dict["last_modified"] = metadata.last_modified.isoformat()
                registry_data[model_id] = metadata_dict

            with open(registry_file, "w") as f:
                json.dump(registry_data, f, indent=2)
            info("Model registry saved successfully")

        except Exception as e:
            error(f"Failed to save model registry", {"error": str(e), "error_details": traceback.format_exc()})

    def register_model(self, model_path: Union[str, Path], metadata: ModelMetadata) -> str:
        """Register a new model in the management system.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the model files (file or directory)
        metadata : ModelMetadata
            Model metadata information

        Returns
        -------
        str
            Generated model ID

        Raises
        ------
        FileNotFoundError
            If model path does not exist
        ValueError
            If model already exists (use force=True to override)
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        if not metadata.model_id:
            metadata.model_id = self._generate_model_id(metadata.model_name)

        metadata.checksum = self._calculate_checksum(model_path)
        metadata.model_size_mb = self._calculate_size(model_path)

        model_storage_path = Path(self.config.MODEL_STORAGE_PATH) / metadata.model_id
        model_storage_path.mkdir(parents=True, exist_ok=True)

        if model_path.is_file():
            shutil.copy2(model_path, model_storage_path)
            metadata.model_files = [model_path.name]
        else:
            for item in model_path.rglob("*"):
                if item.is_file():
                    relative_path = item.relative_to(model_path)
                    dest_path = model_storage_path / relative_path
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)

            metadata.model_files = [
                str(p.relative_to(model_storage_path)) for p in model_storage_path.rglob("*") if p.is_file()
            ]

        metadata.last_modified = datetime.now()
        if not metadata.created_date:
            metadata.created_date = metadata.last_modified

        self.model_registry[metadata.model_id] = metadata
        self._save_registry()

        if metadata.model_id not in self.version_history:
            self.version_history[metadata.model_id] = []

        version = ModelVersion(version_id=metadata.version, changes="Initial model registration")
        self.version_history[metadata.model_id].append(version)

        info(f"Model registered: {metadata.model_id} ({metadata.model_name})")
        return metadata.model_id

    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID.

        Parameters
        ----------
        model_name : str
            Name of the model

        Returns
        -------
        str
            Unique model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_name = "".join(c for c in model_name if c.isalnum() or c in ["_", "-"]).lower()
        return f"{clean_name}_{timestamp}"

    def _calculate_checksum(self, path: Path) -> str:
        """Calculate MD5 checksum for model files.

        Parameters
        ----------
        path : Path
            Path to calculate checksum for

        Returns
        -------
        str
            MD5 checksum
        """
        hash_md5 = hashlib.md5()

        if path.is_file():
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
        else:
            for file_path in sorted(path.rglob("*")):
                if file_path.is_file():
                    with open(file_path, "rb") as f:
                        for chunk in iter(lambda: f.read(4096), b""):
                            hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def _calculate_size(self, path: Path) -> float:
        """Calculate total size in MB.

        Parameters
        ----------
        path : Path
            Path to calculate size for

        Returns
        -------
        float
            Total size in megabytes
        """
        total_size = 0

        if path.is_file():
            total_size = path.stat().st_size
        else:
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size

        return total_size / (1024 * 1024)

    def list_models(self, scenario: Optional[str] = None, status: Optional[str] = None) -> List[ModelMetadata]:
        """List registered models with optional filtering.

        Parameters
        ----------
        scenario : str, optional
            Filter by scenario name
        status : str, optional
            Filter by deployment status

        Returns
        -------
        List[ModelMetadata]
            List of matching models sorted by last modified date
        """
        models = list(self.model_registry.values())

        if scenario:
            models = [m for m in models if m.scenario == scenario]

        if status:
            models = [m for m in models if m.deployment_status == status]

        models.sort(key=lambda m: m.last_modified, reverse=True)

        return models

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID.

        Parameters
        ----------
        model_id : str
            Model ID to retrieve

        Returns
        -------
        Optional[ModelMetadata]
            Model metadata if found, None otherwise
        """
        return self.model_registry.get(model_id)

    def update_model_metadata(self, model_id: str, **updates) -> bool:
        """Update model metadata.

        Parameters
        ----------
        model_id : str
            Model ID to update
        **updates : Any
            Metadata fields to update

        Returns
        -------
        bool
            Success status
        """
        if model_id not in self.model_registry:
            error(f"Model not found", {"model_id": model_id})
            return False

        metadata = self.model_registry[model_id]

        for key, value in updates.items():
            if hasattr(metadata, key):
                setattr(metadata, key, value)

        metadata.last_modified = datetime.now()
        self._save_registry()

        info("Model metadata updated", {"model_id": model_id})
        return True

    def export_model(
        self, model_id: str, export_path: Optional[str] = None, format: str = "tar.gz", include_metadata: bool = True
    ) -> str:
        """Export model to portable format.

        Parameters
        ----------
        model_id : str
            Model ID to export
        export_path : str, optional
            Export destination path
        format : str, optional
            Export format ('tar.gz', 'zip'), by default 'tar.gz'
        include_metadata : bool, optional
            Include metadata in export, by default True

        Returns
        -------
        str
            Path to exported file

        Raises
        ------
        ValueError
            If model not found
        FileNotFoundError
            If model files not found
        """
        if model_id not in self.model_registry:
            raise ValueError(f"Model not found: {model_id}")

        metadata = self.model_registry[model_id]
        model_path = Path(self.config.MODEL_STORAGE_PATH) / model_id

        if not model_path.exists():
            raise FileNotFoundError(f"Model files not found: {model_path}")

        if export_path is None:
            export_dir = Path(self.config.EXPORT_STORAGE_PATH)
            export_filename = f"{metadata.model_name}_{metadata.version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            export_path = export_dir / f"{export_filename}.{format}"
        else:
            export_path = Path(export_path)

        export_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "export"
            temp_path.mkdir()

            model_export_path = temp_path / "model"
            shutil.copytree(model_path, model_export_path)

            if include_metadata:
                metadata_dict = asdict(metadata)
                metadata_dict["created_date"] = metadata.created_date.isoformat()
                metadata_dict["last_modified"] = metadata.last_modified.isoformat()

                with open(temp_path / "metadata.json", "w") as f:
                    json.dump(metadata_dict, f, indent=2)

                if model_id in self.version_history:
                    version_data = []
                    for version in self.version_history[model_id]:
                        version_dict = asdict(version)
                        version_dict["created_date"] = version.created_date.isoformat()
                        version_data.append(version_dict)

                    with open(temp_path / "version_history.json", "w") as f:
                        json.dump(version_data, f, indent=2)

            if format == "tar.gz":
                with tarfile.open(export_path, "w:gz") as tar:
                    tar.add(temp_path, arcname=".")
            elif format == "zip":
                with zipfile.ZipFile(export_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for file_path in temp_path.rglob("*"):
                        if file_path.is_file():
                            arcname = file_path.relative_to(temp_path)
                            zip_file.write(file_path, arcname)
            else:
                raise ValueError(f"Unsupported export format: {format}")

        info(f"Model exported: {model_id} -> {export_path}")
        return str(export_path)

    def import_model(self, import_path: Union[str, Path], model_name: Optional[str] = None, force: bool = False) -> str:
        """Import model from external source.

        Parameters
        ----------
        import_path : Union[str, Path]
            Path to model archive or directory
        model_name : str, optional
            Override model name
        force : bool, optional
            Force import even if model exists, by default False

        Returns
        -------
        str
            Imported model ID

        Raises
        ------
        FileNotFoundError
            If import path does not exist
        ValueError
            If model already exists and force=False
        """
        import_path = Path(import_path)

        if not import_path.exists():
            raise FileNotFoundError(f"Import path does not exist: {import_path}")

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            if import_path.suffix in [".gz", ".tar", ".zip"]:
                if import_path.name.endswith(".tar.gz") or import_path.suffix == ".tar":
                    with tarfile.open(import_path, "r:*") as tar:
                        tar.extractall(temp_path)
                elif import_path.suffix == ".zip":
                    with zipfile.ZipFile(import_path, "r") as zip_file:
                        zip_file.extractall(temp_path)

                model_dir = None
                metadata_file = None

                for item in temp_path.rglob("*"):
                    if item.name == "model" and item.is_dir():
                        model_dir = item
                    elif item.name == "metadata.json":
                        metadata_file = item

                if model_dir is None:
                    for item in temp_path.rglob("*"):
                        if item.is_dir() and any(f.suffix in [".pb", ".h5", ".ckpt"] for f in item.rglob("*")):
                            model_dir = item
                            break

                if model_dir is None:
                    raise ValueError("No model directory found in archive")

            else:
                model_dir = import_path
                metadata_file = import_path / "metadata.json" if (import_path / "metadata.json").exists() else None

            if metadata_file and metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata_dict = json.load(f)

                metadata_dict["created_date"] = datetime.fromisoformat(metadata_dict["created_date"])
                metadata_dict["last_modified"] = datetime.fromisoformat(metadata_dict["last_modified"])

                metadata = ModelMetadata(**metadata_dict)

                if model_name:
                    metadata.model_name = model_name
                    metadata.model_id = self._generate_model_id(model_name)

            else:
                if model_name is None:
                    model_name = f"imported_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                metadata = ModelMetadata(
                    model_name=model_name,
                    model_id=self._generate_model_id(model_name),
                    version="1.0.0",
                    created_date=datetime.now(),
                    last_modified=datetime.now(),
                    algorithm="unknown",
                    neural_architecture="unknown",
                    reward_function="unknown",
                    hyperparameters={},
                    training_episodes=0,
                    training_duration_hours=0.0,
                    scenario="imported",
                    notes="Model imported from external source",
                )

            if not force and metadata.model_id in self.model_registry:
                raise ValueError(f"Model already exists: {metadata.model_id}. Use force=True to override.")

            return self.register_model(model_dir, metadata)

    def upload_to_s3(self, model_id: str, bucket_name: Optional[str] = None, s3_prefix: Optional[str] = None) -> str:
        """Upload model to AWS S3.

        Parameters
        ----------
        model_id : str
            Model ID to upload
        bucket_name : str, optional
            S3 bucket name
        s3_prefix : str, optional
            S3 prefix for organization

        Returns
        -------
        str
            S3 location URI

        Raises
        ------
        RuntimeError
            If AWS S3 client not available
        ValueError
            If model not found
        """
        if self.s3_client is None:
            raise RuntimeError("AWS S3 client not available")

        if model_id not in self.model_registry:
            raise ValueError(f"Model not found: {model_id}")

        metadata = self.model_registry[model_id]

        if bucket_name is None:
            bucket_name = f"{self.config.S3_BUCKET_PREFIX}-{self.aws_session.region_name}"

        if s3_prefix is None:
            s3_prefix = f"models/{metadata.scenario}/{metadata.model_name}"

        export_path = self.export_model(model_id, format="tar.gz")

        try:
            s3_key = f"{s3_prefix}/{Path(export_path).name}"

            self.s3_client.upload_file(export_path, bucket_name, s3_key)

            s3_location = f"s3://{bucket_name}/{s3_key}"

            self.update_model_metadata(model_id, s3_location=s3_location, deployment_status="uploaded")

            info(f"Model uploaded to S3: {s3_location}")
            return s3_location

        finally:
            if Path(export_path).exists():
                Path(export_path).unlink()

    def deploy_to_deepracer(self, model_id: str, deployment_config: AWSDeepRacerConfig) -> str:
        """Deploy model to AWS DeepRacer service.

        Parameters
        ----------
        model_id : str
            Model ID to deploy
        deployment_config : AWSDeepRacerConfig
            Deployment configuration

        Returns
        -------
        str
            Training job ARN

        Raises
        ------
        RuntimeError
            If AWS DeepRacer client not available
        ValueError
            If model not found
        """
        if self.deepracer_client is None:
            raise RuntimeError("AWS DeepRacer client not available")

        if model_id not in self.model_registry:
            raise ValueError(f"Model not found: {model_id}")

        metadata = self.model_registry[model_id]

        if not metadata.s3_location:
            s3_location = self.upload_to_s3(model_id, deployment_config.s3_bucket)
        else:
            s3_location = metadata.s3_location

        try:
            training_job_config = {
                "TrainingJobName": deployment_config.training_job_name
                or f"{metadata.model_name}-{int(datetime.now().timestamp())}",
                "RoleArn": deployment_config.role_arn,
                "AlgorithmSpecification": deployment_config.algorithm_specification,
                "InputDataConfig": [
                    {
                        "ChannelName": "training",
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": s3_location,
                                "S3DataDistributionType": "FullyReplicated",
                            }
                        },
                        "ContentType": "application/x-parquet",
                        "CompressionType": "None",
                    }
                ],
                "OutputDataConfig": {
                    "S3OutputPath": f"s3://{deployment_config.s3_bucket}/{deployment_config.output_s3_key or 'output'}"
                },
                "ResourceConfig": {
                    "InstanceType": deployment_config.instance_type,
                    "InstanceCount": deployment_config.instance_count,
                    "VolumeSizeInGB": deployment_config.volume_size_gb,
                },
                "StoppingCondition": {"MaxRuntimeInSeconds": deployment_config.max_runtime_hours * 3600},
                "HyperParameters": deployment_config.hyperparameters,
            }

            response = self.sagemaker_client.create_training_job(**training_job_config)
            training_job_arn = response["TrainingJobArn"]

            self.update_model_metadata(model_id, deepracer_model_arn=training_job_arn, deployment_status="deployed")

            info(f"Model deployed to AWS DeepRacer: {training_job_arn}")
            return training_job_arn

        except Exception as e:
            error(f"Deployment failed", {"error": str(e), "error_details": traceback.format_exc()})
            raise

    def download_from_s3(self, s3_uri: str, model_name: Optional[str] = None) -> str:
        """Download and import model from S3.

        Parameters
        ----------
        s3_uri : str
            S3 URI of the model
        model_name : str, optional
            Override model name

        Returns
        -------
        str
            Imported model ID

        Raises
        ------
        RuntimeError
            If AWS S3 client not available
        ValueError
            If invalid S3 URI format
        """
        if self.s3_client is None:
            raise RuntimeError("AWS S3 client not available")

        if not s3_uri.startswith("s3://"):
            raise ValueError("Invalid S3 URI format")

        s3_parts = s3_uri[5:].split("/", 1)
        bucket_name = s3_parts[0]
        s3_key = s3_parts[1] if len(s3_parts) > 1 else ""

        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            self.s3_client.download_file(bucket_name, s3_key, temp_path)

            imported_model_id = self.import_model(temp_path, model_name)

            info(f"Model downloaded and imported from S3: {s3_uri}")
            return imported_model_id

        finally:
            if Path(temp_path).exists():
                Path(temp_path).unlink()

    def get_model_statistics(self) -> Dict[str, Any]:
        """Get comprehensive model registry statistics.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing registry statistics
        """
        total_models = len(self.model_registry)
        total_size = sum(m.model_size_mb for m in self.model_registry.values())

        scenarios = {}
        algorithms = {}
        architectures = {}

        for model in self.model_registry.values():
            scenarios[model.scenario] = scenarios.get(model.scenario, 0) + 1
            algorithms[model.algorithm] = algorithms.get(model.algorithm, 0) + 1
            architectures[model.neural_architecture] = architectures.get(model.neural_architecture, 0) + 1

        return {
            "total_models": total_models,
            "total_size_mb": total_size,
            "scenarios": scenarios,
            "algorithms": algorithms,
            "architectures": architectures,
            "registry_path": str(Path(self.config.MODEL_STORAGE_PATH) / "model_registry.json"),
        }
