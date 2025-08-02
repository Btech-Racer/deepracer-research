import time
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, FileSystemLoader

from deepracer_research.config.track.track_type import TrackType
from deepracer_research.deployment.aws_ec2.enum.region import AWSRegion
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario


def get_template_variables(
    model_name: str,
    deployment_target: str,
    s3_bucket: str,
    cloud: str = "local",
    region: str = AWSRegion.US_EAST_1.value,
    dr_world_name: str = TrackType.REINVENT_2019_TRACK.value,
    race_type: str = ExperimentalScenario.TIME_TRIAL.to_deepracer_race_type(),
    h2b_randomize_bot_locations: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Generate template variables for deployment environment files.

    Parameters
    ----------
    model_name : str
        Name of the DeepRacer model
    deployment_target : str
        Target deployment platform (aws_ec2, nvidia_brev, thunder_compute, aws_sagemaker)
    s3_bucket : str
        S3 bucket name
    cloud : str, optional
        Cloud provider, by default "local"
    region : str, optional
        AWS region, by default "us-east-1"
    dr_world_name : str, optional
        DeepRacer world/track name, by default "reInvent2019_track"
    race_type : str, optional
        Race type, by default "TIME_TRIAL"
    h2b_randomize_bot_locations : bool, optional
        Whether to randomize head-to-head bot locations, by default False
    **kwargs
        Additional template variables

    Returns
    -------
    Dict[str, Any]
        Template variables dictionary
    """

    is_object_avoidance = race_type in ["OBJECT_AVOIDANCE", "OBJECT_AVOIDANCE_STATIC", "OBJECT_AVOIDANCE_DYNAMIC"]
    is_head_to_head = race_type in ["HEAD_TO_HEAD", "H2H"]
    is_time_trial = race_type == ExperimentalScenario.TIME_TRIAL.to_deepracer_race_type()

    oa_object_positions = ""
    if is_object_avoidance:
        oa_object_positions = "0.23,-1;0.46,1;0.67,-1;0.85,1"

    h2b_is_lane_change = is_head_to_head
    if not h2b_randomize_bot_locations and is_head_to_head:
        h2b_randomize_bot_locations = True

    base_vars = {
        "model_name": model_name,
        "deployment_target": deployment_target,
        "s3_bucket": s3_bucket,
        "cloud": cloud,
        "region": region,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "workers": kwargs.get("workers", 4),
        "gui_enable": True,
        "host_x": True,
        "display": ":99",
        "robomaker_mount_logs": True,
        "camera_sub_enable": True,
        "eval_save_mp4": True,
        "robomaker_cuda_devices": "0",
        "sagemaker_cuda_devices": "0",
        "local_s3_profile": "minio",
        "kinesis_stream_name": "false",
        "aws_dr_bucket_role": "to-be-defined",
        "simapp_version_tag": "5.3.3-gpu",
        "docker_style": "swarm",
        "dr_world_name": dr_world_name,
        "race_type": race_type,
        "car_name": "FastCar",
        "oa_object_positions": oa_object_positions,
        "h2b_randomize_bot_locations": h2b_randomize_bot_locations,
        "h2b_is_lane_change": h2b_is_lane_change,
        "is_object_avoidance": is_object_avoidance,
        "is_head_to_head": is_head_to_head,
        "is_time_trial": is_time_trial,
    }

    base_vars.update(kwargs)

    return base_vars


def generate_environment_files(
    models_dir: Path,
    template_vars: Dict[str, Any],
    templates_dir: Optional[Path] = None,
    create_individual_workers: bool = True,
) -> Dict[str, Path]:
    """Generate environment files using Jinja2 templates.

    Parameters
    ----------
    models_dir : Path
        Directory to create environment files in
    template_vars : Dict[str, Any]
        Template variables for rendering
    templates_dir : Optional[Path], optional
        Templates directory, by default None (uses deployment/templates)
    create_individual_workers : bool, optional
        Whether to create individual worker-N.env files, by default True

    Returns
    -------
    Dict[str, Path]
        Dictionary mapping file type to created file path
    """
    if templates_dir is None:
        templates_dir = Path(__file__).parent

    jinja_env = Environment(loader=FileSystemLoader(str(templates_dir)))

    created_files = {}

    system_template = jinja_env.get_template("template-system.env.j2")
    system_env_content = system_template.render(**template_vars)
    system_env_path = models_dir / "system.env"
    system_env_path.write_text(system_env_content)
    created_files["system_env"] = system_env_path

    run_template = jinja_env.get_template("template-run.env.j2")
    run_env_content = run_template.render(**template_vars)
    run_env_path = models_dir / "run.env"
    run_env_path.write_text(run_env_content)
    created_files["run_env"] = run_env_path

    worker_template = jinja_env.get_template("template-worker.env.j2")

    num_workers = template_vars.get("workers", 4)

    if create_individual_workers and num_workers > 1:
        for worker_id in range(1, num_workers + 1):
            worker_vars = template_vars.copy()
            worker_vars["worker_id"] = worker_id

            worker_env_content = worker_template.render(**worker_vars)
            worker_env_path = models_dir / f"worker-{worker_id}.env"
            worker_env_path.write_text(worker_env_content)
            created_files[f"worker_{worker_id}_env"] = worker_env_path
    else:
        if num_workers == 1:
            print("Single worker detected: Using swarm configuration (no worker.env file created)")
        else:
            print(f"Same race mode with {num_workers} workers: Using swarm configuration (no worker.env files created)")

    return created_files


def generate_deployment_files_with_s3(
    model_name: str,
    deployment_target: str,
    s3_bucket: Optional[str] = None,
    deepracer_config=None,
    region: str = "us-east-1",
    **template_kwargs,
) -> Dict[str, Any]:
    """Complete workflow to generate deployment files including S3 bucket creation.

    Parameters
    ----------
    model_name : str
        Name of the DeepRacer model
    deployment_target : str
        Target deployment platform
    s3_bucket : Optional[str], optional
        S3 bucket name, by default None (auto-generated)
    deepracer_config : Optional[AWSDeepRacerConfig], optional
        DeepRacer configuration, by default None (creates basic config)
    region : str, optional
        AWS region, by default "us-east-1"
    **template_kwargs
        Additional template variables

    Returns
    -------
    Dict[str, Any]
        Results including models_dir, model_id, s3_bucket, and created_files
    """
    from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager
    from deepracer_research.utils.s3_utils import create_deepracer_s3_bucket

    if not s3_bucket:
        s3_info = create_deepracer_s3_bucket(deployment_target, model_name)
        s3_bucket = s3_info["name"]

    if deepracer_config is None:
        from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig

        deepracer_config = AWSDeepRacerConfig(
            model_name=model_name,
            track_arn=f"arn:aws:deepracer:{region}::track:reInvent2019_track",
            reward_function_code="def reward_function(params): return 1.0",
        )

    deepracer_manager = DeepRacerDeploymentManager(region=region)
    models_dir, model_id = deepracer_manager.create_essential_model_files(deepracer_config)

    template_vars = get_template_variables(
        model_name=model_name, deployment_target=deployment_target, s3_bucket=s3_bucket, region=region, **template_kwargs
    )

    env_files = generate_environment_files(models_dir, template_vars)

    return {
        "models_dir": models_dir,
        "model_id": model_id,
        "s3_bucket": s3_bucket,
        "created_files": env_files,
        "template_vars": template_vars,
    }
