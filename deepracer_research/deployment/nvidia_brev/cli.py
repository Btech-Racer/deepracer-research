import json
import sys
import time

import click

from deepracer_research.deployment.nvidia_brev.api.client import NvidiaBrevClient
from deepracer_research.deployment.nvidia_brev.config.deepracer_config import NvidiaBrevDeepRacerConfig
from deepracer_research.deployment.nvidia_brev.config.nvidia_brev_config import NvidiaBrevConfig
from deepracer_research.deployment.nvidia_brev.enum.deployment_mode import DeploymentMode
from deepracer_research.deployment.nvidia_brev.enum.gpu_type import GPUType
from deepracer_research.deployment.nvidia_brev.enum.instance_template import InstanceTemplate
from deepracer_research.deployment.nvidia_brev.management.deployment_manager import NvidiaBrevDeploymentManager


@click.group()
@click.option("--config-file", type=click.Path(exists=True), help="Configuration file path")
@click.option("--api-token", envvar="NVIDIA_BREV_API_TOKEN", help="NVIDIA Brev API token")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
@click.pass_context
def cli(ctx, config_file, api_token, verbose):
    """NVIDIA Brev CLI for DeepRacer deployments."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["config_file"] = config_file
    ctx.obj["api_token"] = api_token


@cli.command()
@click.option("--model-name", required=True, help="Name for the DeepRacer model")
@click.option("--track-arn", required=True, help="ARN of the track to train on")
@click.option("--reward-function", type=click.File("r"), help="Reward function file")
@click.option("--reward-function-code", help="Reward function code as string")
@click.option("--gpu-type", type=click.Choice([gpu.value for gpu in GPUType]), default=GPUType.A100.value, help="GPU type")
@click.option(
    "--deployment-mode",
    type=click.Choice([mode.value for mode in DeploymentMode]),
    default=DeploymentMode.SPOT.value,
    help="Deployment mode",
)
@click.option("--s3-bucket", help="S3 bucket for model storage (auto-generated if not provided)")
@click.option("--auto-start", is_flag=True, help="Auto-start training after deployment")
@click.option(
    "--files-only",
    is_flag=True,
    help="Generate complete model files (hyperparameters.json, model_metadata.json, reward_function.py) and environment configuration files without creating NVIDIA Brev instance",
)
@click.option("--output", type=click.File("w"), help="Output file for deployment info")
@click.pass_context
def deploy(
    ctx,
    model_name,
    track_arn,
    reward_function,
    reward_function_code,
    gpu_type,
    deployment_mode,
    s3_bucket,
    auto_start,
    files_only,
    output,
):
    """Deploy a DeepRacer training environment."""
    try:
        api_token = ctx.obj.get("api_token")
        if not api_token:
            click.echo("Error: NVIDIA Brev API token is required. Set NVIDIA_BREV_API_TOKEN or use --api-token", err=True)
            sys.exit(1)

        if reward_function:
            reward_function_code = reward_function.read()
        elif not reward_function_code:
            click.echo("Error: Reward function is required. Use --reward-function or --reward-function-code", err=True)
            sys.exit(1)

        config = NvidiaBrevDeepRacerConfig.create_quick_training(
            model_name=model_name,
            track_arn=track_arn,
            reward_function_code=reward_function_code,
            api_token=api_token,
            s3_bucket=s3_bucket,
            gpu_type=GPUType(gpu_type),
            deployment_mode=DeploymentMode(deployment_mode),
            auto_start_training=auto_start,
        )

        if files_only:
            from pathlib import Path

            from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager
            from deepracer_research.utils.s3_utils import create_deepracer_s3_bucket

            if not s3_bucket:
                s3_info = create_deepracer_s3_bucket("nvidia-brev", model_name)
                s3_bucket = s3_info["name"]

            deepracer_manager = DeepRacerDeploymentManager(region="us-east-1")
            models_dir, model_id = deepracer_manager.create_essential_model_files(config.aws_deepracer_config)

            files_dir = models_dir

            from jinja2 import Environment, FileSystemLoader

            template_dir = Path(__file__).parent.parent / "templates"
            jinja_env = Environment(loader=FileSystemLoader(str(template_dir)))

            template_vars = {
                "model_name": model_name,
                "deployment_target": "nvidia_brev",
                "s3_bucket": s3_bucket,
                "cloud": "aws",
                "region": "us-east-1",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "workers": 4,
                "gui_enable": True,
                "host_x": True,
                "display": ":99",
                "robomaker_mount_logs": True,
                "camera_sub_enable": True,
                "eval_save_mp4": True,
                "robomaker_cuda_devices": "0",
                "sagemaker_cuda_devices": "0",
                "dr_world_name": "reInvent2019_track",
                "race_type": "TIME_TRIAL",
                "h2b_randomize_bot_locations": False,
                "h2b_is_lane_change": False,
                "oa_object_positions": "",
                "car_name": "FastCar",
                "local_s3_profile": "default",
                "kinesis_stream_name": "false",
            }

            system_template = jinja_env.get_template("template-system.env.j2")
            system_env_content = system_template.render(**template_vars)
            (files_dir / "system.env").write_text(system_env_content)

            run_template = jinja_env.get_template("template-run.env.j2")
            run_env_content = run_template.render(**template_vars)
            (files_dir / "run.env").write_text(run_env_content)

            worker_template = jinja_env.get_template("template-worker.env.j2")
            worker_env_content = worker_template.render(**template_vars)
            (files_dir / "worker.env").write_text(worker_env_content)

            click.echo(f"\n🎯 NVIDIA Brev Deployment Files Generated!")
            click.echo(f"📦 S3 Bucket: {s3_bucket}")
            click.echo(f"📁 Files Location: {models_dir}")
            click.echo(f"🆔 Model ID: {model_id}")
            click.echo(f"📋 Files created:")
            click.echo(f"   ✅ reward_function.py")
            click.echo(f"   ✅ model_metadata.json")
            click.echo(f"   ✅ hyperparameters.json")
            click.echo(f"   ✅ run.env")
            click.echo(f"   ✅ system.env")
            click.echo(f"   ✅ worker.env")

            if output:
                result_info = {
                    "deployment_type": "nvidia_brev_files",
                    "model_name": model_name,
                    "s3_bucket": s3_bucket,
                    "model_files_dir": str(models_dir),
                    "env_files_dir": str(files_dir),
                    "status": "files_generated",
                    "created_files": [
                        "reward_function.py",
                        "model_metadata.json",
                        "hyperparameters.json",
                        "run.env",
                        "system.env",
                        "worker.env",
                    ],
                }
                json.dump(result_info, output, indent=2)

            return

        manager = NvidiaBrevDeploymentManager(config)

        def progress_callback(message):
            if ctx.obj["verbose"]:
                click.echo(f"[INFO] {message}")

        click.echo("Starting NVIDIA Brev deployment...")
        result = manager.deploy(progress_callback=progress_callback)

        if result["success"]:
            click.echo("✅ Deployment completed successfully!")

            if "access_info" in result:
                click.echo("\n📋 Access Information:")
                for key, value in result["access_info"].items():
                    click.echo(f"  {key}: {value}")

            if output:
                deployment_info = {
                    "model_name": model_name,
                    "instance_id": result.get("instance", {}).get("instance_id"),
                    "access_info": result.get("access_info", {}),
                    "deployment_time": time.time(),
                }
                json.dump(deployment_info, output, indent=2)
                click.echo(f"\n💾 Deployment info saved to {output.name}")

        else:
            click.echo(f"❌ Deployment failed: {result['message']}", err=True)
            if ctx.obj["verbose"] and "error" in result:
                click.echo(f"Error details: {result['error']}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Deployment failed: {str(e)}", err=True)
        if ctx.obj["verbose"]:
            import traceback

            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.option("--instance-id", help="Instance ID to check")
@click.option("--model-name", help="Model name to check")
@click.pass_context
def status(ctx, instance_id, model_name):
    """Check deployment status."""
    try:
        api_token = ctx.obj.get("api_token")
        if not api_token:
            click.echo("Error: NVIDIA Brev API token is required", err=True)
            sys.exit(1)

        client = NvidiaBrevClient(NvidiaBrevConfig(api_token=api_token))

        if instance_id:
            response = client.get_instance(instance_id)
            if response.success and response.instance:
                instance = response.instance
                click.echo(f"Instance: {instance.name}")
                click.echo(f"Status: {instance.status.value}")
                click.echo(f"GPU: {instance.gpu_type.display_name}")
                click.echo(f"Host: {instance.connection_host}")

                if instance.metrics:
                    click.echo(f"Uptime: {instance.metrics.uptime_hours:.1f} hours")
                    if instance.metrics.total_cost:
                        click.echo(f"Cost: ${instance.metrics.total_cost:.2f}")
            else:
                click.echo(f"❌ Instance {instance_id} not found", err=True)
        else:
            instances = client.list_instances()
            if not instances:
                click.echo("No instances found")
                return

            click.echo(f"Found {len(instances)} instances:")
            for instance in instances:
                status_emoji = "🟢" if instance.is_running else "🔴"
                click.echo(f"  {status_emoji} {instance.name} ({instance.instance_id}) - {instance.status.value}")

    except Exception as e:
        click.echo(f"❌ Failed to get status: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--instance-id", required=True, help="Instance ID to destroy")
@click.option("--force", is_flag=True, help="Force destruction without confirmation")
@click.pass_context
def destroy(ctx, instance_id, force):
    """Destroy a deployment."""
    try:
        api_token = ctx.obj.get("api_token")
        if not api_token:
            click.echo("Error: NVIDIA Brev API token is required", err=True)
            sys.exit(1)

        if not force:
            if not click.confirm(f"Are you sure you want to destroy instance {instance_id}?"):
                click.echo("Cancelled")
                return

        client = NvidiaBrevClient(NvidiaBrevConfig(api_token=api_token))

        response = client.delete_instance(instance_id)

        if response.success:
            click.echo(f"✅ Instance {instance_id} destruction initiated")
        else:
            click.echo(f"❌ Failed to destroy instance: {response.message}", err=True)
            sys.exit(1)

    except Exception as e:
        click.echo(f"❌ Failed to destroy instance: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def list_gpu_types(ctx):
    """List available GPU types."""
    click.echo("Available GPU types:")
    for gpu in GPUType:
        suitable = "✅" if gpu.is_suitable_for_training else "⚠️"
        click.echo(f"  {suitable} {gpu.value}: {gpu.display_name} ({gpu.memory_gb}GB)")


@cli.command()
@click.pass_context
def list_templates(ctx):
    """List available instance templates."""
    click.echo("Available templates:")
    for template in InstanceTemplate:
        suitable = "✅" if template.is_suitable_for_deepracer else "⚠️"
        click.echo(f"  {suitable} {template.value}: {template.display_name}")


@cli.command()
@click.option("--gpu-type", type=click.Choice([gpu.value for gpu in GPUType]), default=GPUType.A100.value)
@click.option(
    "--deployment-mode", type=click.Choice([mode.value for mode in DeploymentMode]), default=DeploymentMode.SPOT.value
)
@click.option("--hours", type=int, default=8, help="Number of hours to estimate")
@click.pass_context
def estimate_cost(ctx, gpu_type, deployment_mode, hours):
    """Estimate training costs."""
    try:
        from deepracer_research.deployment.nvidia_brev import get_cost_estimate

        gpu = GPUType(gpu_type)
        mode = DeploymentMode(deployment_mode)

        estimate = get_cost_estimate(gpu, mode, hours)

        click.echo(f"Cost estimate for {gpu.display_name} ({mode.display_name}):")
        click.echo(f"  Hours: {hours}")
        click.echo(f"  Hourly rate: ${estimate['hourly_rate']:.2f}")
        click.echo(f"  Total cost: ${estimate['total_cost']:.2f}")
        click.echo(f"  GPU cost: ${estimate['gpu_cost']:.2f}")

    except Exception as e:
        click.echo(f"❌ Failed to estimate cost: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option("--instance-id", required=True, help="Instance ID to connect to")
@click.option("--local-port", type=int, default=8888, help="Local port for forwarding")
@click.option("--remote-port", type=int, default=8888, help="Remote port to forward")
@click.pass_context
def tunnel(ctx, instance_id, local_port, remote_port):
    """Create SSH tunnel to instance."""
    try:
        api_token = ctx.obj.get("api_token")
        if not api_token:
            click.echo("Error: NVIDIA Brev API token is required", err=True)
            sys.exit(1)

        client = NvidiaBrevClient(NvidiaBrevConfig(api_token=api_token))

        response = client.get_instance(instance_id)
        if not response.success or not response.instance:
            click.echo(f"❌ Instance {instance_id} not found", err=True)
            sys.exit(1)

        instance = response.instance

        if not instance.can_connect:
            click.echo(f"❌ Instance is not ready for connections (status: {instance.status.value})", err=True)
            sys.exit(1)

        from deepracer_research.deployment.nvidia_brev.config.ssh_config import SSHConfig
        from deepracer_research.deployment.nvidia_brev.ssh.ssh_manager import NvidiaBrevSSHManager

        ssh_config = SSHConfig.for_development()
        ssh_manager = NvidiaBrevSSHManager(ssh_config, instance)

        click.echo(f"Creating tunnel: localhost:{local_port} -> {instance.connection_host}:{remote_port}")
        click.echo("Press Ctrl+C to stop...")

        try:
            tunnel_process = ssh_manager.port_forward(local_port, remote_port)

            tunnel_process.wait()

        except KeyboardInterrupt:
            click.echo("\nTunnel stopped")
            if tunnel_process:
                tunnel_process.terminate()

    except Exception as e:
        click.echo(f"❌ Failed to create tunnel: {str(e)}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
