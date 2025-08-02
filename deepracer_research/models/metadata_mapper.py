from typing import Optional

from deepracer_research.config.aws.aws_model_metadata import AWSModelMetadata
from deepracer_research.models.model_metadata import ModelMetadata


def model_metadata_to_aws(research_metadata: ModelMetadata, author: Optional[str] = None) -> AWSModelMetadata:
    """Convert research ModelMetadata to AWS deployment format

    Parameters
    ----------
    research_metadata : ModelMetadata
        Comprehensive research model metadata
    author : str, optional
        Override author field for AWS deployment

    Returns
    -------
    AWSModelMetadata
        AWS-compatible metadata for deployment
    """
    performance_metrics = {
        "completion_rate": research_metadata.completion_rate,
        "best_lap_time": research_metadata.best_lap_time if research_metadata.best_lap_time != float("inf") else None,
        "average_speed": research_metadata.average_speed,
        "training_episodes": research_metadata.training_episodes,
        "training_duration_hours": research_metadata.training_duration_hours,
    }

    performance_metrics = {k: v for k, v in performance_metrics.items() if v is not None}

    tags = {
        "algorithm": research_metadata.algorithm,
        "architecture": research_metadata.neural_architecture,
        "scenario": research_metadata.scenario,
        "track": research_metadata.track_name,
        "version": research_metadata.version,
        "deployment_status": research_metadata.deployment_status,
    }

    if research_metadata.tags:
        for i, tag in enumerate(research_metadata.tags):
            tags[f"tag_{i}"] = tag

    tags = {k: v for k, v in tags.items() if v and v != ""}

    return AWSModelMetadata(
        created_at=research_metadata.created_date.isoformat(),
        author=author,
        experiment_id=research_metadata.experiment_id or None,
        research_phase="training_complete" if research_metadata.convergence_episode > 0 else "training",
        training_duration=f"{research_metadata.training_duration_hours:.2f}h",
        model_size_mb=research_metadata.model_size_mb if research_metadata.model_size_mb > 0 else None,
        performance_metrics=performance_metrics if performance_metrics else None,
        tags=tags if tags else None,
        notes=research_metadata.notes or None,
    )


def aws_metadata_to_model(aws_metadata: AWSModelMetadata, model_name: str, model_id: Optional[str] = None) -> ModelMetadata:
    """Convert AWS metadata back to research ModelMetadata format.

    Parameters
    ----------
    aws_metadata : AWSModelMetadata
        AWS deployment metadata
    model_name : str
        Model name for the research metadata
    model_id : str, optional
        Model ID, if None will use model_name

    Returns
    -------
    ModelMetadata
        Reconstructed research metadata
    """
    from datetime import datetime

    performance = aws_metadata.performance_metrics or {}

    tags_dict = aws_metadata.tags or {}
    algorithm = tags_dict.get("algorithm", "PPO")
    architecture = tags_dict.get("architecture", "CNN_3_layers")
    scenario = tags_dict.get("scenario", "")
    track_name = tags_dict.get("track", "")
    version = tags_dict.get("version", "1.0.0")
    deployment_status = tags_dict.get("deployment_status", "deployed")

    tag_list = []
    for key, value in tags_dict.items():
        if key.startswith("tag_"):
            tag_list.append(value)

    training_duration = 0.0
    if aws_metadata.training_duration:
        try:
            duration_str = aws_metadata.training_duration.replace("h", "")
            training_duration = float(duration_str)
        except (ValueError, AttributeError):
            training_duration = 0.0

    try:
        created_date = datetime.fromisoformat(aws_metadata.created_at.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        created_date = datetime.now()

    return ModelMetadata(
        model_name=model_name,
        model_id=model_id or model_name.replace(" ", "_").lower(),
        version=version,
        created_date=created_date,
        last_modified=created_date,
        algorithm=algorithm,
        neural_architecture=architecture,
        reward_function="aws_deployed",
        training_episodes=performance.get("training_episodes", 0),
        training_duration_hours=training_duration,
        completion_rate=performance.get("completion_rate", 0.0),
        best_lap_time=performance.get("best_lap_time", float("inf")),
        average_speed=performance.get("average_speed", 0.0),
        convergence_episode=performance.get("training_episodes", 0),
        model_size_mb=aws_metadata.model_size_mb or 0.0,
        experiment_id=aws_metadata.experiment_id or "",
        scenario=scenario,
        track_name=track_name,
        notes=aws_metadata.notes or "",
        tags=tag_list,
        deployment_status=deployment_status,
    )


def create_aws_metadata_from_experiment(
    experiment_id: str, scenario: str, author: Optional[str] = None, **kwargs
) -> AWSModelMetadata:
    """Create AWS metadata for a new experiment

    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier
    scenario : str
        Experimental scenario name
    author : str, optional
        Author/researcher name
    **kwargs
        Additional metadata fields

    Returns
    -------
    AWSModelMetadata
        New AWS metadata instance
    """
    tags = {"scenario": scenario, "experiment_type": "research", "framework": "deepracer-research"}
    tags.update(kwargs.get("tags", {}))

    return AWSModelMetadata(
        author=author,
        experiment_id=experiment_id,
        research_phase="experiment_setup",
        tags=tags,
        notes=kwargs.get("notes"),
        **{k: v for k, v in kwargs.items() if k not in ["tags", "notes"]},
    )


def merge_metadata_for_deployment(research_metadata: ModelMetadata, aws_metadata: AWSModelMetadata) -> AWSModelMetadata:
    """Merge research and AWS metadata for deployment

    Parameters
    ----------
    research_metadata : ModelMetadata
        Source research metadata
    aws_metadata : AWSModelMetadata
        AWS deployment metadata (takes priority)

    Returns
    -------
    AWSModelMetadata
        Merged metadata optimized for AWS deployment
    """
    base_aws = model_metadata_to_aws(research_metadata, aws_metadata.author)

    merged_tags = base_aws.tags or {}
    if aws_metadata.tags:
        merged_tags.update(aws_metadata.tags)

    merged_performance = base_aws.performance_metrics or {}
    if aws_metadata.performance_metrics:
        merged_performance.update(aws_metadata.performance_metrics)

    return AWSModelMetadata(
        created_at=aws_metadata.created_at or base_aws.created_at,
        framework=aws_metadata.framework,
        converted_by=aws_metadata.converted_by,
        author=aws_metadata.author or base_aws.author,
        experiment_id=aws_metadata.experiment_id or base_aws.experiment_id,
        research_phase=aws_metadata.research_phase or base_aws.research_phase,
        dataset_version=aws_metadata.dataset_version,
        training_duration=aws_metadata.training_duration or base_aws.training_duration,
        model_size_mb=aws_metadata.model_size_mb or base_aws.model_size_mb,
        performance_metrics=merged_performance if merged_performance else None,
        tags=merged_tags if merged_tags else None,
        notes=aws_metadata.notes or base_aws.notes,
    )
