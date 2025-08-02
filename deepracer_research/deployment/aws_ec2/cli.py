import json
import time
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from deepracer_research.deployment.aws_ec2.config import AWSConfig, EC2InstanceConfig, EC2SSHConfig
from deepracer_research.deployment.aws_ec2.enum import AWSRegion, EC2DeploymentMode, EC2InstanceType
from deepracer_research.deployment.aws_ec2.management import EC2DeploymentError, EC2DeploymentManager
from deepracer_research.deployment.aws_ec2.ssh import EC2SSHManager
from deepracer_research.utils import error

app = typer.Typer(name="ec2-deepracer", help="AWS EC2 DeepRacer deployment CLI", add_completion=False)

console = Console()


def get_aws_config() -> AWSConfig:
    """Get AWS configuration from environment or defaults.

    Returns
    -------
    AWSConfig
        AWS configuration.
    """
    return AWSConfig.from_environment()


def create_manager() -> EC2DeploymentManager:
    """Create and return deployment manager.

    Returns
    -------
    EC2DeploymentManager
        Configured deployment manager instance.
    """
    aws_config = get_aws_config()
    return EC2DeploymentManager(aws_config)


@app.command("list")
def list_instances(include_terminated: bool = typer.Option(False, "--include-terminated", help="Include terminated instances")):
    """List all EC2 instances."""
    try:
        manager = create_manager()
        instances = manager.list_instances(include_terminated=include_terminated)

        if not instances:
            console.print("No instances found.", style="yellow")
            return

        table = Table(title=f"EC2 Instances ({len(instances)} found)")
        table.add_column("Instance ID", style="cyan")
        table.add_column("Instance Type", style="green")
        table.add_column("Status", justify="center")
        table.add_column("IP Address", style="blue")
        table.add_column("Name", style="white")

        for instance in instances:
            status_style = {
                "running": "green",
                "pending": "yellow",
                "stopped": "red",
                "stopping": "yellow",
                "terminated": "red bold",
                "shutting-down": "red dim",
            }.get(instance.status.value, "white")

            table.add_row(
                instance.instance_id,
                instance.instance_type,
                f"[{status_style}]{instance.status.value}[/{status_style}]",
                instance.public_ip or "N/A",
                instance.name or "N/A",
            )

        console.print(table)

    except Exception as e:
        error("Failed to list instances", extra={"error": str(e)})
        console.print(f"❌ Error listing instances: {e}", style="red")
        raise typer.Exit(1)


@app.command("create")
def create_instance(
    preset: Optional[str] = typer.Option(
        None, "--preset", help="Use predefined configuration (training, evaluation, development)"
    ),
    instance_type: str = typer.Option(EC2InstanceType.G4DN_XLARGE.value, "--instance-type", help="EC2 instance type"),
    region: str = typer.Option(AWSRegion.US_EAST_1.value, "--region", help="AWS region"),
    deployment_mode: str = typer.Option(EC2DeploymentMode.ON_DEMAND.value, "--deployment-mode", help="EC2 deployment mode"),
    key_name: Optional[str] = typer.Option(None, "--key-name", help="EC2 key pair name"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="S3 bucket name (auto-generated if not provided)"),
    instance_name: str = typer.Option("deepracer-ec2", "--name", help="Instance name"),
    no_install_deepracer: bool = typer.Option(False, "--no-install-deepracer", help="Skip DeepRacer installation"),
    no_wait: bool = typer.Option(False, "--no-wait", help="Don't wait for instance to be ready"),
    files_only: bool = typer.Option(
        False,
        "--files-only",
        help="Generate complete model files (hyperparameters.json, model_metadata.json, reward_function.py) and environment configuration files without creating EC2 instance",
    ),
):
    """Create a new EC2 instance for DeepRacer."""
    try:
        manager = create_manager()

        try:
            instance_type_enum = EC2InstanceType(instance_type)
        except ValueError:
            console.print(f"❌ Invalid instance type: {instance_type}", style="red")
            valid_types = [t.value for t in EC2InstanceType]
            console.print(f"Available types: {', '.join(valid_types)}")
            raise typer.Exit(1)

        try:
            region_enum = AWSRegion(region)
        except ValueError:
            console.print(f"❌ Invalid region: {region}", style="red")
            valid_regions = [r.value for r in AWSRegion]
            console.print(f"Available regions: {', '.join(valid_regions)}")
            raise typer.Exit(1)

        try:
            deployment_mode_enum = EC2DeploymentMode(deployment_mode)
        except ValueError:
            console.print(f"❌ Invalid deployment mode: {deployment_mode}", style="red")
            valid_modes = [m.value for m in EC2DeploymentMode]
            console.print(f"Available modes: {', '.join(valid_modes)}")
            raise typer.Exit(1)

        if files_only:
            pass

            from deepracer_research.utils.s3_utils import create_deepracer_s3_bucket

            if not s3_bucket:
                s3_info = create_deepracer_s3_bucket("aws-ec2", instance_name)
                s3_bucket = s3_info["name"]

            from deepracer_research.deployment.templates.template_utils import generate_deployment_files_with_s3

            result = generate_deployment_files_with_s3(
                model_name=instance_name, deployment_target="aws_ec2", s3_bucket=s3_bucket, region=region
            )

            models_dir = result["models_dir"]
            model_id = result["model_id"]
            s3_bucket = result["s3_bucket"]

            console.print(f"\n🎯 AWS EC2 Deployment Files Generated!")
            console.print(f"📦 S3 Bucket: {s3_bucket}")
            console.print(f"📁 Files Location: {models_dir}")
            console.print(f"🆔 Model ID: {model_id}")
            console.print(f"📋 Files created:")
            console.print(f"   ✅ reward_function.py")
            console.print(f"   ✅ model_metadata.json")
            console.print(f"   ✅ hyperparameters.json")
            console.print(f"   ✅ run.env")
            console.print(f"   ✅ system.env")
            console.print(f"   ✅ worker.env")
            return

        if preset == "training":
            config = EC2InstanceConfig.for_deepracer_training(
                region=region_enum,
                instance_type=instance_type_enum,
                s3_bucket_name=s3_bucket,
                key_name=key_name,
                instance_name=instance_name,
            )
        elif preset == "evaluation":
            config = EC2InstanceConfig.for_deepracer_evaluation(
                region=region_enum, instance_type=instance_type_enum, key_name=key_name, instance_name=instance_name
            )
        elif preset == "development":
            config = EC2InstanceConfig.for_development(
                region=region_enum, instance_type=instance_type_enum, key_name=key_name, instance_name=instance_name
            )
        else:
            config = EC2InstanceConfig(
                instance_type=instance_type_enum,
                region=region_enum,
                deployment_mode=deployment_mode_enum,
                key_name=key_name,
                instance_name=instance_name,
                s3_bucket_name=s3_bucket,
                install_deepracer_cloud=not no_install_deepracer,
            )

        console.print(f"🚀 Creating EC2 instance...")
        console.print(f"   Instance Type: {instance_type}")
        console.print(f"   Region: {region}")
        console.print(f"   Deployment Mode: {deployment_mode}")
        console.print(f"   Name: {instance_name}")

        if config.install_deepracer_cloud:
            console.print("   DeepRacer-for-Cloud will be installed automatically")
        if no_wait:
            console.print("   Not waiting for instance to be ready")
        if s3_bucket:
            console.print(f"   S3 Bucket: {s3_bucket}")

        with console.status("[bold green]Creating instance (may take 5-10 minutes)..."):
            result = manager.deploy_deepracer_instance(instance_config=config, wait_for_ready=not no_wait)

        if result.success:
            console.print("✅ Instance created successfully!", style="green")
            console.print(f"   Instance ID: {result.instance_id}")
            console.print(f"   Hostname: {result.hostname}")

            if result.ssh_ready:
                console.print("   SSH connection ready")
            if result.deepracer_installed:
                console.print("   DeepRacer-for-Cloud installed and configured")

            instance_file = f"instance_{result.instance_id}.json"
            with open(instance_file, "w") as f:
                json.dump(
                    {
                        "instance_id": result.instance_id,
                        "hostname": result.hostname,
                        "ssh_ready": result.ssh_ready,
                        "deepracer_installed": result.deepracer_installed,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    },
                    f,
                    indent=2,
                )
            console.print(f"   Instance info saved to {instance_file}")

        else:
            console.print(f"❌ Instance creation failed: {result.error_message}", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance creation failed", extra={"error": str(e)})
        console.print(f"❌ Unexpected error: {e}", style="red")
        raise typer.Exit(1)


@app.command("delete")
def delete_instance(instance_id: str):
    """Delete an EC2 instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to delete.
    """
    try:
        manager = create_manager()

        console.print(f"🗑️ Deleting instance {instance_id}...")

        with console.status("[bold red]Deleting instance..."):
            success = manager.delete_instance(instance_id)

        if success:
            console.print("✅ Instance deleted successfully", style="green")
        else:
            console.print("❌ Failed to delete instance", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance deletion failed", extra={"instance_id": instance_id, "error": str(e)})
        console.print(f"❌ Error deleting instance: {e}", style="red")
        raise typer.Exit(1)


@app.command("start")
def start_instance(instance_id: str):
    """Start a stopped EC2 instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to start.
    """
    try:
        manager = create_manager()

        console.print(f"▶️ Starting instance {instance_id}...")

        with console.status("[bold green]Starting instance..."):
            success = manager.start_instance(instance_id)

        if success:
            console.print("✅ Instance started successfully", style="green")
        else:
            console.print("❌ Failed to start instance", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance start failed", extra={"instance_id": instance_id, "error": str(e)})
        console.print(f"❌ Error starting instance: {e}", style="red")
        raise typer.Exit(1)


@app.command("stop")
def stop_instance(instance_id: str):
    """Stop a running EC2 instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to stop.
    """
    try:
        manager = create_manager()

        console.print(f"⏹️ Stopping instance {instance_id}...")

        with console.status("[bold yellow]Stopping instance..."):
            success = manager.stop_instance(instance_id)

        if success:
            console.print("✅ Instance stopped successfully", style="green")
        else:
            console.print("❌ Failed to stop instance", style="red")
            raise typer.Exit(1)

    except Exception as e:
        error("Instance stop failed", extra={"instance_id": instance_id, "error": str(e)})
        console.print(f"❌ Error stopping instance: {e}", style="red")
        raise typer.Exit(1)


@app.command("status")
def show_status(instance_id: str):
    """Show detailed status of an instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to check status for.
    """
    try:
        manager = create_manager()

        console.print(f"Getting status for instance {instance_id}...")

        instance = manager.get_instance_details(instance_id)

        table = Table(title=f"Instance Status: {instance.display_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="white")

        status_style = {
            "running": "green",
            "pending": "yellow",
            "stopped": "red",
            "stopping": "yellow",
            "terminated": "red bold",
            "shutting-down": "red dim",
        }.get(instance.status.value, "white")

        table.add_row("Instance ID", instance.instance_id)
        table.add_row("Status", f"[{status_style}]{instance.status.value}[/{status_style}]")
        table.add_row("Instance Type", instance.instance_type.value)
        table.add_row("Region", instance.region.value)
        table.add_row("Resources", instance.resource_summary)
        table.add_row("Public IP", instance.public_ip or "N/A")
        table.add_row("Private IP", instance.private_ip or "N/A")
        table.add_row("Key Name", instance.key_name or "N/A")
        table.add_row("Launch Time", str(instance.launch_time) if instance.launch_time else "N/A")
        table.add_row("SSH Ready", "Yes" if instance.is_ready_for_ssh else "No")
        table.add_row("Billable", "Yes" if instance.is_billable else "No")
        table.add_row("Hourly Cost", f"${instance.cost_estimate_hourly:.2f}")

        console.print(table)

    except Exception as e:
        error("Failed to get instance status", extra={"instance_id": instance_id, "error": str(e)})
        console.print(f"❌ Error getting status: {e}", style="red")
        raise typer.Exit(1)


@app.command("ssh")
def ssh_to_instance(
    instance_id: str, private_key_path: Optional[str] = typer.Option(None, "--key", help="Path to private key file")
):
    """Open SSH connection to EC2 instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID.
    private_key_path : str, optional
        Path to private key file.
    """
    try:
        console.print(f"Connecting to instance {instance_id}...")

        manager = create_manager()
        instance = manager.get_instance_details(instance_id)

        if not instance.is_ready_for_ssh:
            console.print("❌ Instance is not ready for SSH connection", style="red")
            console.print(f"Status: {instance.status.value}")
            raise typer.Exit(1)

        ssh_config = EC2SSHConfig.for_ec2_default()
        if private_key_path:
            ssh_config.private_key_path = private_key_path
        elif instance.key_name:
            ssh_config.private_key_path = f"~/.ssh/{instance.key_name}.pem"

        ssh_manager = EC2SSHManager(instance_id, instance.public_ip, ssh_config)

        console.print("Opening SSH shell...")
        ssh_manager.open_shell()

    except Exception as e:
        error("SSH connection failed", extra={"error": str(e)})
        console.print(f"❌ SSH connection failed: {e}", style="red")
        raise typer.Exit(1)


@app.command("start-training")
def start_training(instance_id: str, name: str = typer.Option("cli-training", "--name", help="Training session name")):
    """Start DeepRacer training on an instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to start training on.
    name : str
        Name for the training session.
    """
    try:
        manager = create_manager()

        console.print(f"Starting training '{name}' on instance {instance_id}...")

        with console.status("[bold green]Starting training..."):
            success = manager.start_training(instance_id, training_name=name)

        if success:
            console.print(f"✅ Training '{name}' started successfully", style="green")
        else:
            console.print("❌ Failed to start training", style="red")
            raise typer.Exit(1)

    except EC2DeploymentError as e:
        error("Training start failed", extra={"instance_id": instance_id, "training_name": name, "error": str(e)})
        console.print(f"❌ Training error: {e}", style="red")
        raise typer.Exit(1)


@app.command("stop-training")
def stop_training(instance_id: str):
    """Stop DeepRacer training on an instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to stop training on.
    """
    try:
        manager = create_manager()

        console.print(f"Stopping training on instance {instance_id}...")

        with console.status("[bold yellow]Stopping training..."):
            success = manager.stop_training(instance_id)

        if success:
            console.print("✅ Training stopped successfully", style="green")
        else:
            console.print("⚠️ Training may not have been running", style="yellow")

    except Exception as e:
        error("Training stop failed", extra={"instance_id": instance_id, "error": str(e)})
        console.print(f"❌ Error stopping training: {e}", style="red")
        raise typer.Exit(1)


@app.command("logs")
def get_logs(instance_id: str):
    """Get training logs from an instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to get logs from.
    """
    try:
        manager = create_manager()

        console.print(f"Fetching training logs from {instance_id}...")

        logs = manager.get_training_logs(instance_id)

        console.print(f"\n[bold]Training logs for instance {instance_id}:[/bold]")
        console.print("=" * 50)
        console.print(logs)

    except Exception as e:
        error("Failed to fetch logs", extra={"instance_id": instance_id, "error": str(e)})
        console.print(f"❌ Error fetching logs: {e}", style="red")
        raise typer.Exit(1)


@app.command("exec")
def execute_command(instance_id: str, command: str):
    """Execute a command on an instance.

    Parameters
    ----------
    instance_id : str
        EC2 instance ID to execute command on.
    command : str
        Command to execute on the instance.
    """
    try:
        manager = create_manager()

        console.print(f"Executing command on {instance_id}: {command}")

        output = manager.execute_command(instance_id, command)

        console.print(f"\n[bold]Command output from {instance_id}:[/bold]")
        console.print("=" * 50)
        console.print(output)

    except Exception as e:
        error("Command execution failed", extra={"instance_id": instance_id, "command": command, "error": str(e)})
        console.print(f"❌ Error executing command: {e}", style="red")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
