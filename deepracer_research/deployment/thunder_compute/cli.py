import json
import os
import time
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.table import Table

from deepracer_research.config.aws.types.sensor_type import SensorType
from deepracer_research.config.track.track_type import TrackType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.deployment.thunder_compute.config.instance_config import InstanceConfig
from deepracer_research.deployment.thunder_compute.config.ssh_config import SSHConfig
from deepracer_research.deployment.thunder_compute.config.thunder_compute_config import ThunderComputeConfig
from deepracer_research.deployment.thunder_compute.enum.gpu_type import GPUType
from deepracer_research.deployment.thunder_compute.enum.instance_template import InstanceTemplate
from deepracer_research.deployment.thunder_compute.management.deployment_manager import (
    ThunderDeploymentError,
    ThunderDeploymentManager,
)
from deepracer_research.deployment.thunder_compute.ssh.manager import SSHManager
from deepracer_research.utils import error

app = typer.Typer(name="thunder-deepracer", help="Thunder Compute DeepRacer deployment CLI", add_completion=False)

console = Console()


def get_api_token() -> str:
    """Get Thunder Compute API token from environment.

    Returns
    -------
    str
        API token for Thunder Compute authentication.

    Raises
    ------
    typer.Exit
        If API token is not set in environment.
    """
    token = os.getenv("THUNDER_API_TOKEN")
    if not token:
        console.print(" THUNDER_API_TOKEN environment variable not set", style="red")
        console.print("Set it with: export THUNDER_API_TOKEN=your_token_here")
        raise typer.Exit(1)
    return token


def create_manager() -> ThunderDeploymentManager:
    """Create and return deployment manager.

    Returns
    -------
    ThunderDeploymentManager
        Configured deployment manager instance.
    """
    token = get_api_token()
    config = ThunderComputeConfig(api_token=token)
    return ThunderDeploymentManager(config)


@app.command("list-algorithms")
def list_algorithms():
    """List all available training algorithms with descriptions."""
    console.print("🧠 Available Training Algorithms", style="bold blue")
    console.print()

    table = Table(title="Training Algorithms")
    table.add_column("Algorithm", style="cyan", width=15)
    table.add_column("Description", style="white", width=50)
    table.add_column("Use Case", style="green", width=35)
    table.add_column("AWS Support", justify="center", width=12)

    aws_supported = TrainingAlgorithm.get_aws_supported()

    for algorithm in TrainingAlgorithm:
        aws_support_style = "green" if algorithm in aws_supported else "red"
        aws_support_text = "✅ Yes" if algorithm in aws_supported else "❌ No"

        table.add_row(
            algorithm.value,
            algorithm.get_description(),
            algorithm.get_use_case(),
            f"[{aws_support_style}]{aws_support_text}[/{aws_support_style}]",
        )

    console.print(table)
    console.print()

    console.print("📊 Algorithm Categories:", style="bold yellow")
    console.print(f"   🟢 AWS Supported: {', '.join([alg.value for alg in TrainingAlgorithm.get_aws_supported()])}")
    console.print(f"   🎮 Continuous Control: {', '.join([alg.value for alg in TrainingAlgorithm.get_continuous_control()])}")
    console.print(f"   🎯 Discrete Control: {', '.join([alg.value for alg in TrainingAlgorithm.get_discrete_control()])}")
    console.print(f"   👥 Multi-Agent: {', '.join([alg.value for alg in TrainingAlgorithm.get_multi_agent()])}")
    console.print(f"   🔬 Research: {', '.join([alg.value for alg in TrainingAlgorithm.get_research_algorithms()])}")
    console.print()

    console.print("💡 Usage Tips:", style="bold green")
    console.print("   • Use --algorithm <algorithm_name> with deploy-training command")
    console.print("   • AWS-supported algorithms work with AWS DeepRacer service")
    console.print("   • Research algorithms are for advanced experimentation")
    console.print("   • CLIPPED_PPO is the recommended default for most use cases")


@app.command("list")
def list_instances():
    """List all Thunder Compute instances."""
    try:
        manager = create_manager()
        instances = manager.list_instances()

        if not instances:
            console.print("No instances found.", style="yellow")
            return

        table = Table(title=f"Thunder Compute Instances ({len(instances)} found)")
        table.add_column("Identifier", style="cyan")
        table.add_column("UUID", style="dim")
        table.add_column("Status", justify="center")
        table.add_column("Resources", style="green")
        table.add_column("IP Address", style="blue")

        for instance in instances:
            status_style = {
                "running": "green",
                "creating": "yellow",
                "stopped": "red",
                "failed": "red bold",
                "deleting": "red dim",
            }.get(instance.status.value, "white")

            table.add_row(
                instance.identifier,
                instance.uuid[:8] + "...",
                f"[{status_style}]{instance.status.value}[/{status_style}]",
                instance.resource_summary,
                instance.ip_address or "N/A",
            )

        console.print(table)

    except Exception as e:
        error("Failed to list instances", extra={"error": str(e)})
        console.print(f" Error listing instances: {e}", style="red")
        raise typer.Exit(1)


@app.command("create")
def create_instance(
    preset: Optional[str] = typer.Option(
        None, "--preset", help="Use predefined configuration (training, evaluation, research)"
    ),
    cpu_cores: int = typer.Option(8, "--cpu-cores", help="Number of CPU cores"),
    gpu_type: str = typer.Option(GPUType.A100_XL, "--gpu-type", help="GPU type (t4, a100, a100-xl)"),
    disk_size: int = typer.Option(100, "--disk-size", help="Disk size in GB"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="S3 bucket name"),
    no_install_deepracer: bool = typer.Option(False, "--no-install-deepracer", help="Skip DeepRacer installation"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait for instance to be ready"),
):
    """Create a new Thunder Compute instance."""
    try:
        manager = create_manager()

        try:
            gpu_type_enum = GPUType(gpu_type.lower())
        except ValueError:
            console.print(f" Invalid GPU type: {gpu_type}", style="red")
            valid_types = [g.value for g in GPUType]
            console.print(f"Available types: {', '.join(valid_types)}")
            raise typer.Exit(1)

        if preset == "training":
            config = InstanceConfig.for_deepracer_training(
                cpu_cores=cpu_cores, gpu_type=gpu_type_enum, disk_size_gb=disk_size, s3_bucket_name=s3_bucket
            )
        elif preset == "evaluation":
            config = InstanceConfig.for_deepracer_evaluation(
                cpu_cores=cpu_cores, gpu_type=gpu_type_enum, disk_size_gb=disk_size
            )
        elif preset == "research":
            config = InstanceConfig.for_research(cpu_cores=cpu_cores, gpu_type=gpu_type_enum, disk_size_gb=disk_size)
        else:
            config = InstanceConfig(
                cpu_cores=cpu_cores,
                template=InstanceTemplate.BASE,
                gpu_type=gpu_type_enum,
                disk_size_gb=disk_size,
                install_deepracer_cloud=not no_install_deepracer,
            )

        console.print(f"Creating instance with {cpu_cores} CPU cores, {gpu_type} GPU, {disk_size}GB disk...")
        if config.install_deepracer_cloud:
            console.print("DeepRacer-for-Cloud will be installed automatically")
        if no_wait:
            console.print("Not waiting for instance to be ready")

        if s3_bucket:
            console.print(f"Creating S3 bucket: {s3_bucket}")
            from deepracer_research.deployment.thunder_compute.ssh.aws_setup import create_s3_bucket_locally

            bucket_created = create_s3_bucket_locally(s3_bucket)
            if bucket_created:
                console.print(f"S3 bucket {s3_bucket} ready")
            else:
                console.print(f"Warning: Could not create S3 bucket {s3_bucket}", style="yellow")

        with console.status("[bold green]Creating instance (may take 2-3 minutes)..."):
            result = manager.deploy_deepracer_instance(instance_config=config, wait_for_ready=not no_wait)

        if result.success:
            console.print("✅ Instance created successfully!", style="green")
            console.print(f"   UUID: {result.instance_uuid}")
            if result.ssh_ready:
                console.print("   SSH connection ready")
            if result.deepracer_installed:
                console.print("   DeepRacer-for-Cloud installed and configured")

            instance_file = f"instance_{result.instance_uuid[:8]}.json"
            with open(instance_file, "w") as f:
                json.dump(
                    {
                        "uuid": result.instance_uuid,
                        "ssh_ready": result.ssh_ready,
                        "deepracer_installed": result.deepracer_installed,
                    },
                    f,
                    indent=2,
                )
            console.print(f"   Instance info saved to {instance_file}")

        else:
            console.print(f" Instance creation failed: {result.error_message}", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance creation failed", extra={"error": str(e)})
        console.print(f" Unexpected error: {e}", style="red")
        raise typer.Exit(1)


@app.command("delete")
def delete_instance(instance_uuid: str):
    """Delete a Thunder Compute instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to delete.
    """
    try:
        manager = create_manager()

        console.print(f"🗑️ Deleting instance {instance_uuid[:8]}...")

        with console.status("[bold red]Deleting instance..."):
            success = manager.delete_instance(instance_uuid)

        if success:
            console.print("✅ Instance deleted successfully", style="green")
        else:
            console.print(" Failed to delete instance", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance deletion failed", extra={"instance_uuid": instance_uuid, "error": str(e)})
        console.print(f" Error deleting instance: {e}", style="red")
        raise typer.Exit(1)


@app.command("start")
def start_instance(instance_uuid: str):
    """Start a stopped Thunder Compute instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to start.
    """
    try:
        from deepracer_research.deployment.thunder_compute.api.client import ThunderComputeClient

        api_token = get_api_token()
        client = ThunderComputeClient(api_token)

        console.print(f"▶️ Starting instance {instance_uuid[:8]}...")

        with console.status("[bold green]Starting instance..."):
            success = client.start_instance(instance_uuid)

        if success:
            console.print("Instance started successfully", style="green")
        else:
            console.print("Failed to start instance", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance start failed", extra={"instance_uuid": instance_uuid, "error": str(e)})
        console.print(f"Error starting instance: {e}", style="red")
        raise typer.Exit(1)


@app.command("stop")
def stop_instance(instance_uuid: str):
    """Stop a running Thunder Compute instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to stop.
    """
    try:
        from deepracer_research.deployment.thunder_compute.api.client import ThunderComputeClient

        api_token = get_api_token()
        client = ThunderComputeClient(api_token)

        console.print(f"⏹️ Stopping instance {instance_uuid[:8]}...")

        with console.status("[bold yellow]Stopping instance..."):
            success = client.stop_instance(instance_uuid)

        if success:
            console.print("Instance stopped successfully", style="green")
        else:
            console.print("Failed to stop instance", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance stop failed", extra={"instance_uuid": instance_uuid, "error": str(e)})
        console.print(f"Error stopping instance: {e}", style="red")
        raise typer.Exit(1)


@app.command("start-training")
def start_training(instance_uuid: str, name: str = typer.Option("cli-training", "--name", help="Training session name")):
    """Start DeepRacer training on an instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to start training on.
    name : str
        Name for the training session.
    """
    try:
        manager = create_manager()

        console.print(f"Starting training '{name}' on instance {instance_uuid[:8]}...")

        with console.status("[bold green]Starting training..."):
            success = manager.start_training(instance_uuid=instance_uuid, training_name=name)

        if success:
            console.print(f"Training '{name}' started successfully", style="green")
        else:
            console.print(" Failed to start training", style="red")
            raise typer.Exit(1)

    except ThunderDeploymentError as e:
        error("Training start failed", extra={"instance_uuid": instance_uuid, "training_name": name, "error": str(e)})
        console.print(f" Training error: {e}", style="red")
        raise typer.Exit(1)


@app.command("stop-training")
def stop_training(instance_uuid: str):
    """Stop DeepRacer training on an instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to stop training on.
    """
    try:
        manager = create_manager()

        console.print(f"Stopping training on instance {instance_uuid[:8]}...")

        with console.status("[bold yellow]Stopping training..."):
            success = manager.stop_training(instance_uuid)

        if success:
            console.print("Training stopped successfully", style="green")
        else:
            console.print("Training may not have been running", style="yellow")

    except Exception as e:
        error("Training stop failed", extra={"instance_uuid": instance_uuid, "error": str(e)})
        console.print(f" Error stopping training: {e}", style="red")
        raise typer.Exit(1)


@app.command("logs")
def get_logs(instance_uuid: str):
    """Get training logs from an instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to get logs from.
    """
    try:
        manager = create_manager()

        console.print(f"Fetching training logs from {instance_uuid[:8]}...")

        logs = manager.get_training_logs(instance_uuid)

        console.print(f"\n[bold]Training logs for instance {instance_uuid[:8]}:[/bold]")
        console.print("=" * 50)
        console.print(logs)

    except Exception as e:
        error("Failed to fetch logs", extra={"instance_uuid": instance_uuid, "error": str(e)})
        console.print(f" Error fetching logs: {e}", style="red")
        raise typer.Exit(1)


@app.command("exec")
def execute_command(instance_uuid: str, command: str):
    """Execute a command on an instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to execute command on.
    command : str
        Command to execute on the instance.
    """
    try:
        manager = create_manager()

        console.print(f"Executing command on {instance_uuid[:8]}: {command}")

        output = manager.execute_command(instance_uuid, command)

        console.print(f"\n[bold]Command output from {instance_uuid[:8]}:[/bold]")
        console.print("=" * 50)
        console.print(output)

    except Exception as e:
        error("Command execution failed", extra={"instance_uuid": instance_uuid, "command": command, "error": str(e)})
        console.print(f" Error executing command: {e}", style="red")
        raise typer.Exit(1)


@app.command("status")
def show_status(instance_uuid: str):
    """Show detailed status of an instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the instance to check status for.
    """
    try:
        manager = create_manager()

        console.print(f"Getting status for instance {instance_uuid[:8]}...")

        instance = manager.get_instance_details(instance_uuid)

        table = Table(title=f"Instance Status: {instance.display_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        status_style = {
            "running": "green",
            "creating": "yellow",
            "stopped": "red",
            "failed": "red bold",
            "deleting": "red dim",
        }.get(instance.status.value, "white")

        table.add_row("UUID", instance.uuid)
        table.add_row("Status", f"[{status_style}]{instance.status.value}[/{status_style}]")
        table.add_row("Resources", instance.resource_summary)
        table.add_row("Template", instance.template)
        table.add_row("IP Address", instance.ip_address or "N/A")
        table.add_row("Created", instance.created_at or "N/A")
        table.add_row("SSH Ready", "Yes" if instance.is_ready_for_ssh else "No")
        table.add_row("Billable", "Yes" if instance.is_billable else "No")

        console.print(table)

    except Exception as e:
        error("Failed to get instance status", extra={"instance_uuid": instance_uuid, "error": str(e)})
        console.print(f" Error getting status: {e}", style="red")
        raise typer.Exit(1)


@app.command("ssh")
def ssh_to_instance(instance_uuid: str):
    """Open SSH connection to Thunder Compute instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the Thunder Compute instance.
    """
    try:
        console.print(f"Connecting to instance {instance_uuid[:8]}...")

        from deepracer_research.deployment.thunder_compute.api.client import ThunderComputeClient

        api_token = get_api_token()
        client = ThunderComputeClient(api_token)
        instance = client.get_instance(instance_uuid)

        ssh_config = SSHConfig(use_tnr_cli=True)
        ssh_manager = SSHManager(instance_uuid, ssh_config, thunder_cli_index=instance.thunder_cli_index)

        if ssh_manager.setup_tnr_connection():
            console.print("Opening SSH shell...")
            ssh_manager.open_shell()
        else:
            console.print(" Failed to setup TNR connection", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("SSH connection failed", extra={"error": str(e)})
        console.print(f" SSH connection failed: {e}", style="red")
        raise typer.Exit(1)


@app.command("sync")
def sync_project(
    instance_uuid: str,
    project_path: Optional[str] = typer.Option(
        None, "--project-path", help="Local project path (defaults to current directory)"
    ),
    remote_path: str = typer.Option("~/deepracer-research", "--remote-path", help="Remote project path"),
    exclude: Optional[List[str]] = typer.Option(None, "--exclude", help="Additional exclude patterns"),
):
    """Sync DeepRacer project code to Thunder Compute instance.

    Parameters
    ----------
    instance_uuid : str
        UUID of the Thunder Compute instance.
    project_path : str, optional
        Local project path to sync.
    remote_path : str
        Remote destination path.
    exclude : List[str], optional
        Additional patterns to exclude from sync.
    """
    try:
        import os

        if project_path is None:
            project_path = os.getcwd()

        project_path = os.path.abspath(project_path)

        if not os.path.exists(project_path):
            console.print(f" Project path not found: {project_path}", style="red")
            raise typer.Exit(1)

        console.print(f"Syncing project from {project_path}")
        console.print(f"Target: {instance_uuid[:8]}:{remote_path}")

        from deepracer_research.deployment.thunder_compute.api.client import ThunderComputeClient

        api_token = get_api_token()
        client = ThunderComputeClient(api_token)
        instance = client.get_instance(instance_uuid)

        ssh_config = SSHConfig(use_tnr_cli=True)
        ssh_manager = SSHManager(instance_uuid, ssh_config, thunder_cli_index=instance.thunder_cli_index)

        if not ssh_manager.setup_tnr_connection():
            console.print(" Failed to setup TNR connection", style="red")
            raise typer.Exit(1)

        console.print("Waiting for instance to be ready...")
        if not ssh_manager.wait_for_instance_ready(timeout=120):
            console.print(" Instance not ready for SSH", style="red")
            raise typer.Exit(1)

        with console.status("[bold green]Syncing project files..."):
            success = ssh_manager.sync_project(project_root=project_path, remote_project_dir=remote_path)

        if success:
            console.print("Project sync completed successfully!", style="green")
            console.print(f"   Files uploaded to: {remote_path}")
            console.print(f"   Connect with: thunder-compute ssh {instance_uuid}")
        else:
            console.print(" Project sync failed", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Project sync failed", extra={"error": str(e)})
        console.print(f" Project sync failed: {e}", style="red")
        raise typer.Exit(1)


@app.command("upload")
def upload_directory(
    instance_id: str,
    local_path: str,
    remote_path: str,
    exclude: Optional[List[str]] = typer.Option(None, "--exclude", help="Patterns to exclude"),
):
    """Upload directory to Thunder Compute instance.

    Parameters
    ----------
    instance_id : str
        ID of the Thunder Compute instance (e.g., "0", "1", etc.).
    local_path : str
        Local directory path to upload.
    remote_path : str
        Remote destination path.
    exclude : List[str], optional
        Patterns to exclude from upload.
    """
    try:
        import os

        if not os.path.exists(local_path):
            console.print(f" Local path not found: {local_path}", style="red")
            raise typer.Exit(1)

        console.print(f"Uploading {local_path}")
        console.print(f"Target: instance-{instance_id}:{remote_path}")

        from deepracer_research.deployment.thunder_compute.api.client import ThunderComputeClient

        api_token = get_api_token()
        client = ThunderComputeClient(api_token)
        instance = client.get_instance_by_id(instance_id)

        ssh_config = SSHConfig(use_tnr_cli=False)
        ssh_manager = SSHManager(instance.uuid, ssh_config, thunder_cli_index=instance.thunder_cli_index)

        if not ssh_manager.setup_tnr_connection():
            console.print(" Failed to setup SSH connection", style="red")
            raise typer.Exit(1)

        with console.status("[bold green]Uploading files..."):
            success = ssh_manager.upload_directory(local_dir=local_path, remote_dir=remote_path, exclude_patterns=exclude)

        if success:
            console.print("Upload completed successfully!", style="green")
        else:
            console.print("Upload failed", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Upload failed", extra={"error": str(e)})
        console.print(f" Upload failed: {e}", style="red")
        raise typer.Exit(1)


@app.command("deploy-training")
def deploy_training(
    model_id: str = typer.Argument(..., help="Model ID to deploy for training"),
    cpu_cores: int = typer.Option(8, "--cpu-cores", help="Number of CPU cores"),
    gpu_type: str = typer.Option(GPUType.A100_XL, "--gpu-type", help="GPU type (t4, a100, a100-xl)"),
    disk_size: int = typer.Option(100, "--disk-size", help="Disk size in GB"),
    workers: int = typer.Option(1, "--workers", help="Number of workers (1-3) for multi-worker training"),
    same_race_workers: bool = typer.Option(
        True,
        "--same-race/--different-race",
        help="All workers use same race configuration (--same-race) or different configurations (--different-race)",
    ),
    race_type: str = typer.Option(
        "TIME_TRIAL",
        "--race-type",
        help="Race type: TIME_TRIAL, OBJECT_AVOIDANCE, HEAD_TO_HEAD, CENTERLINE_FOLLOWING, SPEED_OPTIMIZATION",
    ),
    randomize_obstacles: bool = typer.Option(
        False,
        "--randomize-obstacles/--static-obstacles",
        help="Randomize obstacle locations vs static positions (only for OBJECT_AVOIDANCE race type)",
    ),
    num_obstacles: int = typer.Option(6, "--num-obstacles", help="Number of obstacles (1-6) for OBJECT_AVOIDANCE race type"),
    obstacle_distance: float = typer.Option(
        2.0, "--obstacle-distance", help="Minimum distance between obstacles in meters (only for OBJECT_AVOIDANCE race type)"
    ),
    bot_car_obstacles: bool = typer.Option(
        False, "--bot-car-obstacles", help="Use bot cars as obstacles instead of boxes (only for OBJECT_AVOIDANCE race type)"
    ),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="S3 bucket name (auto-generated if not provided)"),
    reward_function_file: Optional[Path] = typer.Option(
        None, "--reward-function", "-r", help="Path to reward function file (required for --files-only)"
    ),
    track_name: str = typer.Option(
        TrackType.REINVENT_2019_TRACK.value,
        "--track",
        help="Track name for model configuration. Examples: reInvent2019_track (default), reinvent_base, 2022_april_open, bowtie_track. Use 'thunder-compute list-tracks' to see all available tracks.",
    ),
    sensors: str = typer.Option(
        SensorType.FRONT_FACING_CAMERA.value,
        "--sensors",
        help="Comma-separated sensor types: FRONT_FACING_CAMERA (recommended), SECTOR_LIDAR, LIDAR, STEREO_CAMERAS. Example: 'FRONT_FACING_CAMERA' or 'SECTOR_LIDAR,FRONT_FACING_CAMERA'",
    ),
    algorithm: str = typer.Option(
        TrainingAlgorithm.CLIPPED_PPO.value,
        "--algorithm",
        help=f"Training algorithm to use. Available options: {', '.join([alg.value for alg in TrainingAlgorithm])}. Use 'thunder-compute list-algorithms' to see detailed descriptions and AWS compatibility.",
    ),
    project_path: Optional[str] = typer.Option(
        None, "--project-path", help="Local project path (defaults to current directory)"
    ),
    files_only: bool = typer.Option(
        False,
        "--files-only",
        help="Generate complete model files (hyperparameters.json, model_metadata.json, reward_function.py) and environment configuration files without creating Thunder Compute instance",
    ),
):
    """Deploy DeepRacer training with complete bootstrap workflow.

    Worker Configuration
    --------------------
    - --same-race (default): All workers use swarm configuration (only run.env created)
    - --different-race: Each worker gets different race config (worker-1.env, worker-2.env, etc.)

    Race Type Configuration
    -----------------------
    - Object avoidance mode is automatically enabled when --race-type=OBJECT_AVOIDANCE
    - Use --num-obstacles, --randomize-obstacles, etc. to configure object avoidance

    Algorithm Configuration
    -----------------------
    - Available algorithms: PPO, CLIPPED_PPO (default), SAC, TD3, RAINBOW_DQN, MAML, MADDPG, DREAMER_V2
    - AWS-supported algorithms: PPO, CLIPPED_PPO, SAC
    - Use --algorithm to specify the training algorithm

    Use --files-only to generate complete model files (hyperparameters.json,
    model_metadata.json, reward_function.py) and environment configuration files
    (system.env, run.env, worker-X.env) without creating a Thunder Compute instance.
    This is useful for preparing all necessary files for manual deployment or testing.
    """
    try:
        import os
        from pathlib import Path

        try:
            training_algorithm = TrainingAlgorithm(algorithm)
        except ValueError:
            console.print(f" Invalid algorithm: {algorithm}", style="red")
            console.print(f"Available algorithms: {', '.join([alg.value for alg in TrainingAlgorithm])}")
            console.print(
                f"AWS-supported algorithms: {', '.join([alg.value for alg in TrainingAlgorithm.get_aws_supported()])}"
            )
            console.print("Use 'thunder-compute list-algorithms' for detailed descriptions", style="blue")
            raise typer.Exit(1)

        if project_path is None:
            project_path = os.getcwd()

        project_path = Path(project_path).resolve()

        if not project_path.exists():
            console.print(f" Project path not found: {project_path}", style="red")
            raise typer.Exit(1)

        if not files_only:
            model_dir = project_path / "models" / model_id
            if not model_dir.exists():
                console.print(f" Model directory not found: {model_dir}", style="red")
                console.print(
                    "Generate model files first with: deepracer-research deploy deepracer <model_name> --files-only",
                    style="blue",
                )
                raise typer.Exit(1)

        if workers < 1 or workers > 4:
            console.print(f" Invalid number of workers: {workers}. Must be between 1 and 3.", style="red")
            raise typer.Exit(1)

        valid_race_types = ["TIME_TRIAL", "OBJECT_AVOIDANCE", "HEAD_TO_HEAD", "CENTERLINE_FOLLOWING", "SPEED_OPTIMIZATION"]
        if race_type not in valid_race_types:
            console.print(f" Invalid race type: {race_type}. Must be one of: {', '.join(valid_race_types)}", style="red")
            raise typer.Exit(1)

        try:
            sensor_types = SensorType.parse_sensor_list(sensors)
        except ValueError as e:
            console.print(f" Invalid sensor configuration: {e}", style="red")
            raise typer.Exit(1)

        object_avoidance = race_type == "OBJECT_AVOIDANCE"

        if files_only and reward_function_file is None:
            console.print(f" --reward-function is required when using --files-only", style="red")
            console.print("Specify a reward function file with: --reward-function path/to/reward_function.py", style="blue")
            raise typer.Exit(1)

        if files_only and reward_function_file and not reward_function_file.exists():
            console.print(f" Reward function file not found: {reward_function_file}", style="red")
            raise typer.Exit(1)

        if object_avoidance:
            if num_obstacles < 1 or num_obstacles > 6:
                console.print(f" Invalid number of obstacles: {num_obstacles}. Must be between 1 and 6.", style="red")
                raise typer.Exit(1)
            if obstacle_distance < 0.5 or obstacle_distance > 10.0:
                console.print(
                    f" Invalid obstacle distance: {obstacle_distance}. Must be between 0.5 and 10.0 meters.", style="red"
                )
                raise typer.Exit(1)

        if files_only:
            console.print(f"📁 Generating DeepRacer files for Thunder Compute", style="bold blue")
        else:
            console.print(f"🚀 Starting DeepRacer deployment with bootstrap", style="bold blue")
        console.print(f"📋 Configuration:")
        console.print(f"   Model ID: {model_id}")
        console.print(f"   CPU Cores: {cpu_cores}")
        console.print(f"   GPU Type: {gpu_type}")
        console.print(f"   Disk Size: {disk_size}GB")
        console.print(f"   Workers: {workers}")
        console.print(f"   Race Type: {race_type}")
        console.print(f"   Track: {track_name}")
        console.print(f"   Algorithm: {training_algorithm.value} ({training_algorithm.get_description()})")
        console.print(f"   Sensors: {', '.join([s.value for s in sensor_types])}")
        console.print(f"   Project Path: {project_path}")
        if files_only and reward_function_file:
            console.print(f"   Reward Function: {reward_function_file}")

        if s3_bucket:
            console.print(f"   S3 Bucket: {s3_bucket}")
        else:
            console.print(f"   S3 Bucket: Auto-generated")

        if workers > 1:
            console.print(f"   Multi-worker training enabled with {workers} workers")
            console.print(f"   Workers will use different tracks for varied training")

        if object_avoidance:
            console.print(f"   🚧 Object Avoidance Configuration:", style="bold yellow")
            console.print(f"      • Number of obstacles: {num_obstacles}")
            console.print(f"      • Obstacle positioning: {'Random' if randomize_obstacles else 'Static'}")
            console.print(f"      • Minimum distance: {obstacle_distance}m")
            console.print(f"      • Obstacle type: {'Bot cars' if bot_car_obstacles else 'Boxes'}")

        if race_type == "HEAD_TO_HEAD":
            console.print(f"   🏎️ Head-to-Head Racing Configuration:", style="bold cyan")
            console.print(f"      • 🎲 Randomized bot car parameters for varied training")
            console.print(f"      • Lane changes: Disabled for consistent racing")
            console.print(f"      • Bot car locations: Static positions")

        console.print(f"   🎬 MP4 Recording: Enabled (all training sessions will be recorded)", style="bold green")

        if files_only:
            console.print(f"\n📁 Creating files for Thunder Compute deployment...", style="bold blue")

            bucket_name_only = False
            if not s3_bucket:
                s3_bucket = f"deepracer-{model_id}-{int(time.time())}"
                bucket_name_only = True

            oa_config = None
            if object_avoidance:
                oa_config = {
                    "race_type": "OBJECT_AVOIDANCE",
                    "num_obstacles": num_obstacles,
                    "randomize_obstacles": randomize_obstacles,
                    "obstacle_distance": obstacle_distance,
                    "bot_car_obstacles": bot_car_obstacles,
                }
            else:
                oa_config = {"race_type": race_type}

            try:
                manager = create_manager()

                files_dir = project_path / "thunder_files" / model_id
                files_dir.mkdir(parents=True, exist_ok=True)

                console.print(f"📋 Creating essential model files...")

                with open(reward_function_file, "r") as f:
                    reward_function_code = f.read()

                from deepracer_research.config.track.track_type import get_track_arn_by_name
                from deepracer_research.deployment import AWSDeepRacerConfig
                from deepracer_research.deployment.deepracer import DeepRacerDeploymentManager

                try:
                    track_arn = get_track_arn_by_name(track_name, "us-east-1")
                except ValueError:
                    track_arn = get_track_arn_by_name(TrackType.REINVENT_2019_TRACK.value, "us-east-1")
                    console.print(
                        f"⚠️ Track '{track_name}' not found, using '{TrackType.REINVENT_2019_TRACK.value}'", style="yellow"
                    )

                if training_algorithm == TrainingAlgorithm.SAC:
                    console.print(f"⚠️  SAC algorithm detected - forcing continuous action space", style="yellow")
                    console.print(
                        f"   SAC (Soft Actor-Critic) is designed for continuous control and cannot work with discrete action spaces",
                        style="blue",
                    )
                    console.print(f"   For discrete action spaces, consider using PPO or CLIPPED_PPO instead", style="blue")

                from deepracer_research.experiments import ExperimentalScenario

                scenario_mapping = {
                    "TIME_TRIAL": ExperimentalScenario.TIME_TRIAL,
                    "OBJECT_AVOIDANCE": ExperimentalScenario.OBJECT_AVOIDANCE,
                    "HEAD_TO_HEAD": ExperimentalScenario.HEAD_TO_HEAD,
                    "CENTERLINE_FOLLOWING": ExperimentalScenario.CENTERLINE_FOLLOWING,
                    "SPEED_OPTIMIZATION": ExperimentalScenario.SPEED_OPTIMIZATION,
                }
                scenario = scenario_mapping.get(oa_config["race_type"], ExperimentalScenario.TIME_TRIAL)

                if oa_config["race_type"] == "TIME_TRIAL" and scenario == ExperimentalScenario.TIME_TRIAL:
                    pass

                    model_name_lower = model_id.lower()

                    if "centerline" in model_name_lower or "center-line" in model_name_lower:
                        scenario = ExperimentalScenario.CENTERLINE_FOLLOWING
                        console.print(
                            f"🎯 Auto-detected CENTERLINE_FOLLOWING scenario from model name '{model_id}'", style="cyan"
                        )
                        console.print(f"   Use --race-type CENTERLINE_FOLLOWING to be explicit", style="blue")
                    elif "speed" in model_name_lower and (
                        "optimization" in model_name_lower or "optimisation" in model_name_lower
                    ):
                        scenario = ExperimentalScenario.SPEED_OPTIMIZATION
                        console.print(
                            f"🎯 Auto-detected SPEED_OPTIMIZATION scenario from model name '{model_id}'", style="cyan"
                        )
                        console.print(f"   Use --race-type SPEED_OPTIMIZATION to be explicit", style="blue")

                if scenario in [ExperimentalScenario.OBJECT_AVOIDANCE, ExperimentalScenario.HEAD_TO_HEAD]:
                    if training_algorithm == TrainingAlgorithm.SAC:
                        console.print(
                            f"⚠️  {scenario.value.upper()} + SAC algorithm detected - forcing continuous action space",
                            style="yellow",
                        )
                        console.print(
                            f"   SAC (Soft Actor-Critic) requires continuous action space for all scenarios", style="blue"
                        )
                    else:
                        console.print(
                            f"⚠️  {scenario.value.upper()} scenario detected - recommending continuous action space",
                            style="yellow",
                        )
                        console.print(
                            f"   {scenario.value.replace('_', ' ').title()} scenarios work best with continuous action space",
                            style="blue",
                        )
                        console.print(
                            f"   For {training_algorithm.value}, both discrete and continuous action spaces are supported",
                            style="blue",
                        )

                action_space_type = AWSDeepRacerConfig.get_compatible_action_space_type(training_algorithm, scenario)

                console.print(f"🎛️  Applying optimized action space for {scenario.value}:", style="bold blue")

                scenario_config_preview = {
                    ExperimentalScenario.TIME_TRIAL: {
                        "steering": "±28° (reduced from ±30° for high-speed stability)",
                        "speed": "1.0-4.0 m/s (higher minimum for time optimization)",
                    },
                    ExperimentalScenario.SPEED_OPTIMIZATION: {
                        "steering": "±25° (minimal for straight-line speed)",
                        "speed": "1.5-4.0 m/s (high minimum speed)",
                    },
                    ExperimentalScenario.OBJECT_AVOIDANCE: {
                        "steering": "±30° (full range for obstacle navigation)",
                        "speed": "0.5-2.5 m/s (conservative for safety)",
                    },
                    ExperimentalScenario.HEAD_TO_HEAD: {
                        "steering": "±30° (full range for overtaking)",
                        "speed": "0.8-3.8 m/s (balanced for competition)",
                    },
                    ExperimentalScenario.CENTERLINE_FOLLOWING: {
                        "steering": "±20° (constrained for precision)",
                        "speed": "0.8-3.0 m/s (moderate range)",
                    },
                }.get(scenario, {})

                if scenario_config_preview:
                    console.print(f"   • Steering: {scenario_config_preview['steering']}")
                    console.print(f"   • Speed: {scenario_config_preview['speed']}")
                    console.print(f"   • Based on racing research and community best practices", style="dim")

                deepracer_config = AWSDeepRacerConfig.create_for_scenario(
                    model_name=model_id,
                    track_arn=track_arn,
                    reward_function_code=reward_function_code,
                    training_algorithm=training_algorithm,
                    sensor_type=sensor_types,
                    action_space_type=action_space_type,
                    experimental_scenario=scenario,
                    max_job_duration_seconds=7200,
                )

                if not deepracer_config.validate_configuration():
                    console.print(" Invalid DeepRacer configuration", style="red")
                    raise typer.Exit(1)

                deepracer_deployment_manager = DeepRacerDeploymentManager(region="us-east-1")
                created_models_dir, created_model_id = deepracer_deployment_manager.create_essential_model_files(
                    deepracer_config, output_dir=project_path
                )

                console.print(f"✅ Essential model files created in: {created_models_dir}")

                run_id = f"{created_model_id}-{int(time.time())}"
                files_dir = project_path / "thunder_files" / created_model_id

                if files_dir.exists():
                    for file in files_dir.glob("*.env"):
                        file.unlink()
                        console.print(f"🗑️ Removed existing file: {file.name}")

                files_dir.mkdir(parents=True, exist_ok=True)

                if bucket_name_only:
                    console.print("📦 Creating S3 bucket for deployment...")
                    try:
                        from deepracer_research.utils.s3_utils import create_deepracer_s3_bucket

                        s3_info = create_deepracer_s3_bucket("thunder-compute", created_model_id)
                        s3_bucket = s3_info["name"]
                        console.print(f"✅ S3 bucket successfully created: {s3_bucket}")
                        console.print(f"🌐 S3 URI: {s3_info['uri']}")
                    except Exception as e:
                        console.print(f"❌ Failed to create S3 bucket: {e}", style="red")
                        console.print("⚠️ Continuing with local bucket name for configuration files", style="yellow")
                        s3_bucket = f"deepracer-thunder-compute-{created_model_id}-{int(time.time())}"

                from deepracer_research.deployment.templates.template_utils import (
                    generate_environment_files,
                    get_template_variables,
                )

                template_vars = get_template_variables(
                    model_name=created_model_id,
                    deployment_target="thunder_compute",
                    s3_bucket=s3_bucket,
                    cloud="local",
                    region="us-east-1",
                    dr_world_name=track_name,
                    race_type=race_type,
                    h2b_randomize_bot_locations=race_type == "HEAD_TO_HEAD",
                    workers=workers,
                )

                template_vars["s3_bucket"] = s3_bucket

                create_individual_workers = workers > 1 and not same_race_workers

                env_files = generate_environment_files(
                    models_dir=files_dir, template_vars=template_vars, create_individual_workers=create_individual_workers
                )

                actual_files = [f for f in files_dir.glob("*.env")]
                console.print(f"📁 Thunder Compute configuration files:")
                for file_path in sorted(actual_files):
                    console.print(f"      • {file_path.name}")
                if (files_dir / "deployment_info.json").exists():
                    console.print(f"      • deployment_info.json")

                if workers == 1:
                    console.print(
                        f"🔗 Single worker configured in swarm mode (only run.env + system.env created)", style="blue"
                    )
                elif workers > 1:
                    if same_race_workers:
                        console.print(
                            f"👥 {workers} workers configured in same race swarm mode (only run.env + system.env created)",
                            style="blue",
                        )
                    else:
                        console.print(
                            f"👥 {workers} workers configured with different race settings (individual worker-N.env files created)",
                            style="blue",
                        )

                if workers == 1 or same_race_workers:
                    worker_mode = "swarm"
                else:
                    worker_mode = "different_race"

                deployment_info = {
                    "model_id": created_model_id,
                    "run_id": run_id,
                    "workers": workers,
                    "worker_mode": worker_mode,
                    "race_type": race_type,
                    "s3_bucket": s3_bucket,
                    "deployment_type": "thunder_compute_files",
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "files_location": str(files_dir),
                }

                if object_avoidance:
                    deployment_info["object_avoidance"] = {
                        "num_obstacles": num_obstacles,
                        "randomize_obstacles": randomize_obstacles,
                        "obstacle_distance": obstacle_distance,
                        "bot_car_obstacles": bot_car_obstacles,
                    }

                (files_dir / "deployment_info.json").write_text(json.dumps(deployment_info, indent=2))

                console.print(f"✅ Thunder Compute files created successfully!", style="green")
                console.print(f"📁 Thunder files location: {files_dir}")
                console.print(f"📁 Model files location: {created_models_dir}")
                console.print(f"🆔 Model ID: {model_id}")
                console.print(f"🏃 Run ID: {run_id}")
                console.print(f"🪣 S3 Bucket: {s3_bucket}")
                console.print(f"📋 Files created:")
                console.print(f"   Thunder Compute configuration files:")
                console.print(f"      • system.env")
                console.print(f"      • run.env")
                if workers > 1 and not same_race_workers:
                    for worker_num in range(1, workers + 1):
                        console.print(f"      • worker-{worker_num}.env")
                console.print(f"      • deployment_info.json")
                console.print(f"   Essential model files:")
                console.print(f"      • hyperparameters.json")
                console.print(f"      • model_metadata.json")
                console.print(f"      • reward_function.py")
                console.print(f"\nNext steps:")
                console.print(f"   • Review generated files in: {files_dir}")
                console.print(f"   • Upload model files from: {created_models_dir}")
                console.print(
                    f"   • Deploy with Thunder Compute: thunder-compute deploy-training {model_id} --s3-bucket {s3_bucket}"
                )
                console.print(f"   • Upload files manually to your DeepRacer for Cloud instance")

                return

            except Exception as e:
                console.print(f" Failed to create Thunder Compute files: {e}", style="red")
                import traceback

                traceback.print_exc()
                raise typer.Exit(1)

        try:
            gpu_type_enum = GPUType(gpu_type.lower())
        except ValueError:
            console.print(f" Invalid GPU type: {gpu_type}", style="red")
            valid_types = [g.value for g in GPUType]
            console.print(f"Available types: {', '.join(valid_types)}")
            raise typer.Exit(1)

        manager = create_manager()

        from deepracer_research.deployment.thunder_compute.config.instance_config import InstanceConfig

        instance_config = InstanceConfig.for_deepracer_training(
            cpu_cores=cpu_cores, gpu_type=gpu_type_enum, disk_size_gb=disk_size, s3_bucket_name=s3_bucket
        )

        console.print("\n⏳ Starting deployment (this may take 10-15 minutes)...")

        oa_config = None
        if object_avoidance:
            oa_config = {
                "race_type": "OBJECT_AVOIDANCE",
                "num_obstacles": num_obstacles,
                "randomize_obstacles": randomize_obstacles,
                "obstacle_distance": obstacle_distance,
                "bot_car_obstacles": bot_car_obstacles,
            }
        else:
            oa_config = {"race_type": race_type}

        with console.status("[bold green]Deploying DeepRacer training environment..."):
            result = manager.deploy_deepracer_with_bootstrap(
                model_id=model_id,
                instance_config=instance_config,
                s3_bucket_name=s3_bucket,
                local_project_root=project_path,
                workers=workers,
                object_avoidance_config=oa_config,
            )

            if isinstance(result, tuple):
                instance_uuid, run_id = result
            else:
                instance_uuid = result
                run_id = f"{model_id}-{int(time.time())}"

        console.print("✅ DeepRacer deployment completed successfully!", style="green")
        console.print(f"🆔 Instance UUID: {instance_uuid}")
        console.print(f"📋 Model ID: {model_id}")
        console.print(f"🏃 Run ID: {run_id}")

        console.print("\nNext steps:", style="bold yellow")
        console.print(f"   • Monitor training: thunder-compute logs {instance_uuid}")
        console.print(f"   • SSH to instance: thunder-compute ssh {instance_uuid}")
        console.print(f"   • Stop training: thunder-compute stop-training {instance_uuid}")
        console.print(f"   • Delete instance: thunder-compute delete {instance_uuid}")

        instance_file = f"training_{model_id}_{instance_uuid[:8]}.json"
        with open(instance_file, "w") as f:
            json.dump(
                {
                    "model_id": model_id,
                    "instance_uuid": instance_uuid,
                    "run_id": run_id,
                    "deployment_type": "bootstrap_training",
                    "workers": workers,
                },
                f,
                indent=2,
            )
        console.print(f"   • Deployment info saved to: {instance_file}")

        runs_path = project_path / "runs" / run_id
        console.print(f"   • Run configuration saved to: {runs_path}")

        if workers > 1:
            console.print(f"   • Multi-worker setup with {workers} workers configured")
            console.print(f"   • Workers will train on different tracks for diversity")

        console.print(f"\n🎯 Instance UUID for future reference: {instance_uuid}", style="bold cyan")

    except Exception as e:
        error("DeepRacer deployment failed", extra={"error": str(e)})
        console.print(f" Deployment failed: {e}", style="red")
        raise typer.Exit(1)


@app.command()
def list_tracks():
    """List all available DeepRacer tracks."""
    console = Console()

    console.print("🏁 Available DeepRacer Tracks", style="bold blue")
    console.print()

    track_categories = {
        "Current Competition Tracks": [
            TrackType.REINVENT_2024_CHAMP,
            TrackType.REINVENT_2022_CHAMP,
        ],
        "2019 Tracks (Recommended)": [
            TrackType.REINVENT_2019_TRACK,
            TrackType.REINVENT_BASE,
            TrackType.REINVENT_2019_WIDE,
        ],
        "2022 Monthly Tracks": [
            TrackType.APRIL_2022_OPEN,
            TrackType.MAY_2022_OPEN,
            TrackType.JUNE_2022_OPEN,
            TrackType.JULY_2022_OPEN,
            TrackType.AUGUST_2022_OPEN,
            TrackType.SEPTEMBER_2022_OPEN,
            TrackType.OCTOBER_2022_OPEN,
        ],
        "Classic Training Tracks": [
            TrackType.BOWTIE_TRACK,
            TrackType.OVAL_TRACK,
            TrackType.CANADA_TRAINING,
            TrackType.TOKYO_TRAINING_TRACK,
        ],
        "International Tracks": [
            TrackType.CHINA_TRACK,
            TrackType.MEXICO_TRACK,
            TrackType.NEW_YORK_TRACK,
            TrackType.VEGAS_TRACK,
            TrackType.SPAIN_TRACK,
        ],
    }

    for category, tracks in track_categories.items():
        console.print(f"📂 {category}", style="bold yellow")
        for track in tracks:
            marker = "⭐" if track == TrackType.REINVENT_2019_TRACK else "  "
            console.print(f"  {marker} {track.value}")
        console.print()

    console.print(f"Default track: {TrackType.REINVENT_2019_TRACK.value} (marked with ⭐)", style="blue")
    console.print("Use any track value with: --track <track_name>", style="blue")


if __name__ == "__main__":
    app()
