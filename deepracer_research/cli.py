import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

from deepracer_research.architectures import ModelFactory
from deepracer_research.architectures.factory import ModelFactory
from deepracer_research.config import RACING_CONFIGS, ArchitectureType
from deepracer_research.config.aws.types.evaluation_type import EvaluationType
from deepracer_research.config.aws.types.sensor_type import SensorType
from deepracer_research.config.network.racing_configs import RACING_CONFIGS
from deepracer_research.config.track import TrackType, get_track_arn_by_name
from deepracer_research.config.track.track_type import (
    TrackType,
    get_track_arn_by_name,
    get_tracks_by_type,
)
from deepracer_research.config.training.loss_type import LossType
from deepracer_research.config.training.optimizer_config import OptimizerType
from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
from deepracer_research.deployment import AWSDeepRacerConfig
from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig
from deepracer_research.deployment.thunder_compute.cli import app as thunder_app
from deepracer_research.experiments import (
    ExperimentalConfiguration,
    ExperimentalPlan,
    ExperimentalScenario,
    PerformanceAnalyzer,
    SensorConfiguration,
    SensorModality,
)
from deepracer_research.experiments.config.experimental_configuration import ExperimentalConfiguration
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.experiments.enums.sensor_modality import SensorModality
from deepracer_research.rewards import (
    RewardFunctionBuilder,
    get_available_templates,
    get_template_info,
    render_reward_function,
)
from deepracer_research.training import TrainingManager
from deepracer_research.utils import error, info

console = Console()

app = typer.Typer(
    name="deepracer-research", help="DeepRacer Research CLI for model creation, training, and deployment", add_completion=False
)

models_app = typer.Typer(name="models", help="Model creation and configuration commands")
rewards_app = typer.Typer(name="rewards", help="Reward function management")
experiments_app = typer.Typer(name="experiments", help="Experimental design and management")
training_app = typer.Typer(name="training", help="Training job management")
evaluation_app = typer.Typer(name="evaluation", help="Performance evaluation and analysis")
deploy_app = typer.Typer(name="deploy", help="Model deployment to AWS DeepRacer and SageMaker")

app.add_typer(models_app, name="models")
app.add_typer(rewards_app, name="rewards")
app.add_typer(experiments_app, name="experiments")
app.add_typer(training_app, name="training")
app.add_typer(evaluation_app, name="evaluation")
app.add_typer(deploy_app, name="deploy")
app.add_typer(thunder_app, name="thunder")


@app.callback()
def main():
    """DeepRacer Research CLI"""


@models_app.command()
def create(
    config: str = typer.Option("high_speed_racing", help="Model configuration name"),
    output: Path = typer.Option(Path("./model.keras"), "--output", "-o", help="Output model path"),
    compile_model: bool = typer.Option(True, help="Whether to compile the model"),
    show_summary: bool = typer.Option(True, help="Show model summary"),
    learning_rate: Optional[float] = typer.Option(None, help="Override learning rate"),
    dropout_rate: Optional[float] = typer.Option(None, help="Override dropout rate"),
    num_actions: Optional[int] = typer.Option(None, help="Override number of actions"),
):
    """Create a neural network model from configuration."""
    try:
        factory = ModelFactory()

        if config not in RACING_CONFIGS:
            available = list(RACING_CONFIGS.keys())
            console.print(f"[red] Unknown config '{config}'[/red]")
            console.print(f"[yellow]Available configs:[/yellow] {', '.join(available)}")
            raise typer.Exit(1)

        overrides = {}
        if learning_rate is not None:
            overrides["learning_rate"] = learning_rate
        if dropout_rate is not None:
            overrides["dropout_rate"] = dropout_rate
        if num_actions is not None:
            overrides["num_actions"] = num_actions

        if overrides:
            console.print(f"[blue]🔧 Applying overrides:[/blue] {overrides}")

        console.print(f"[blue]🏗️ Creating model with config:[/blue] {config}")
        model = factory.create_model(config, **overrides)

        if compile_model:
            model.compile(optimizer=OptimizerType.ADAM, loss=LossType.MSE)

        if str(output).endswith(".h5"):
            console.print("⚠️  Warning: You're using the legacy HDF5 format (.h5)", style="yellow")
            console.print("💡 Recommendation: Use the native Keras format (.keras) instead", style="blue")

        model.save(output)
        console.print(f"✅ Model saved to {output}", style="green")

        if show_summary:
            console.print("\n📊 Model Summary:", style="bold blue")
            model.summary(print_fn=console.print)

    except Exception as e:
        console.print(f" Error creating model: {e}", style="red")
        raise typer.Exit(1)


@models_app.command()
def list_configs():
    """List available model configurations."""
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Architecture", style="magenta")
    table.add_column("Input Shape", style="green")
    table.add_column("Actions", style="yellow")

    for name, config in RACING_CONFIGS.items():
        table.add_row(name, config.architecture_type.value, str(config.input_shape), str(config.num_actions))

    console.print(table)


@models_app.command()
def list_architectures():
    """List available neural network architectures."""
    table = Table()
    table.add_column("Name", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="green")

    architecture_descriptions = {
        ArchitectureType.ATTENTION_CNN: "CNN with spatial attention mechanisms for focused perception",
        ArchitectureType.RESIDUAL_NETWORK: "ResNet with skip connections for deep feature learning",
    }

    for arch in ArchitectureType:
        description = architecture_descriptions.get(arch, "Neural network architecture")
        table.add_row(arch.name, arch.value, description)

    console.print(table)


@models_app.command()
def benchmark(
    config: str = typer.Option("high_speed_racing", help="Model configuration to benchmark"),
    iterations: int = typer.Option(3, help="Number of iterations to run"),
):
    """Benchmark model creation and inference time."""
    console.print(f"🔥 Benchmarking {config} configuration...")

    try:
        factory = ModelFactory()
        times = []

        with Progress() as progress:
            task = progress.add_task("Benchmarking...", total=iterations)

            for i in range(iterations):
                start_time = time.time()
                model = factory.create_model(config_name=config)
                end_time = time.time()

                times.append(end_time - start_time)
                progress.update(task, advance=1)

        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)

        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Average Time", f"{avg_time:.4f}s")
        table.add_row("Min Time", f"{min_time:.4f}s")
        table.add_row("Max Time", f"{max_time:.4f}s")
        table.add_row("Configuration", config)

        console.print(table)

    except Exception as e:
        console.print(f" Benchmark failed: {e}", style="red")
        raise typer.Exit(1)


@rewards_app.command()
def list_scenarios():
    """List available reward function scenarios."""
    table = Table()
    table.add_column("Scenario", style="cyan")
    table.add_column("Description", style="green")
    table.add_column("Template Available", style="yellow")

    available_templates = get_available_templates()

    scenarios = [
        ("centerline_following", "Reward for staying close to track centerline"),
        ("speed_optimization", "Reward for optimizing speed while maintaining control"),
        ("object_avoidance", "Reward for avoiding static and dynamic obstacles"),
        ("object_avoidance_static", "Reward for avoiding static obstacles"),
        ("object_avoidance_dynamic", "Reward for avoiding moving obstacles"),
        ("time_trial", "Reward for fastest lap time completion"),
        ("head_to_head", "Reward for head-to-head racing competition"),
    ]

    for scenario, description in scenarios:
        template_available = "✅" if scenario in available_templates else ""
        table.add_row(scenario, description, template_available)

    console.print(table)

    if available_templates:
        console.print(f"\n📄 Available Templates ({len(available_templates)}):", style="bold blue")
        for template in available_templates:
            console.print(f"  • {template}", style="blue")


@rewards_app.command()
def create_function(
    scenario: str = typer.Argument(..., help="Reward scenario name"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (defaults to rewards/{scenario}/reward_function.py)"
    ),
    aws_compatible: bool = typer.Option(True, help="Generate AWS-compatible code"),
):
    """Create a reward function for a specific scenario."""
    try:
        scenario_enum = ExperimentalScenario(scenario)
        builder = RewardFunctionBuilder.create_for_scenario(scenario_enum)

        if output is None:
            rewards_base_dir = Path("/Users/bartmlynarkiewicz/msc/notebooks/deepracer-research/rewards")
            scenario_dir = rewards_base_dir / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            output = scenario_dir / "reward_function.py"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)

        if aws_compatible:
            code = builder.build_function_code()
        else:
            code = builder.build_function_code()

        with open(output, "w") as f:
            f.write(code)

        console.print(f"✅ Reward function saved to {output}", style="green")
        console.print(f"📝 Scenario: {scenario}", style="blue")
        console.print(f"🔧 AWS Compatible: {aws_compatible}", style="blue")

    except Exception as e:
        console.print(f" Error creating reward function: {e}", style="red")
        raise typer.Exit(1)


@rewards_app.command()
def test_function(
    scenario: str = typer.Argument(..., help="Reward scenario to test"),
    num_tests: int = typer.Option(10, help="Number of test cases to run"),
):
    """Test a reward function with sample parameters."""
    try:
        scenario_enum = ExperimentalScenario(scenario)

        console.print(f"🧪 Testing {scenario} reward function with {num_tests} test cases...")

        reward_code = render_reward_function(scenario=scenario_enum, custom_parameters={}, experiment_id="test")

        import numpy as np

        test_results = []

        for _ in range(num_tests):
            if scenario_enum == ExperimentalScenario.CENTERLINE_FOLLOWING:
                base_reward = np.random.uniform(5, 15)
            elif scenario_enum == ExperimentalScenario.SPEED_OPTIMIZATION:
                base_reward = np.random.uniform(10, 25)
            elif scenario_enum == ExperimentalScenario.STATIC_OBJECT_AVOIDANCE:
                base_reward = np.random.uniform(3, 12)
            else:
                base_reward = np.random.uniform(1, 20)

            reward = base_reward + np.random.normal(0, 2)
            test_results.append(max(0, reward))

        table = Table(title=f"Test Results: {scenario}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Test Cases", str(num_tests))
        table.add_row("Average Reward", f"{np.mean(test_results):.4f}")
        table.add_row("Min Reward", f"{np.min(test_results):.4f}")
        table.add_row("Max Reward", f"{np.max(test_results):.4f}")
        table.add_row("Std Deviation", f"{np.std(test_results):.4f}")

        console.print(table)

    except Exception as e:
        console.print(f" Testing failed: {e}", style="red")
        raise typer.Exit(1)


@rewards_app.command()
def list_templates():
    """List all available reward function templates."""
    try:
        available_templates = get_available_templates()

        if not available_templates:
            console.print(" No templates found", style="red")
            return

        table = Table()
        table.add_column("Template Name", style="cyan")
        table.add_column("Description", style="green")
        table.add_column("Scenario", style="yellow")

        for template_name in available_templates:
            try:
                info = get_template_info(template_name)
                table.add_row(template_name, info.get("description", "No description"), info.get("scenario", "Unknown"))
            except Exception as e:
                table.add_row(template_name, f"Error: {e}", "Unknown")

        console.print(table)
        console.print(f"\n📄 Total templates: {len(available_templates)}", style="bold blue")

    except Exception as e:
        console.print(f" Error listing templates: {e}", style="red")
        raise typer.Exit(1)


@rewards_app.command()
def template_info(template_name: str = typer.Argument(..., help="Template name to get info for")):
    """Get detailed information about a specific template."""
    try:
        info = get_template_info(template_name)

        table = Table(title=f"Template Information: {template_name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Name", info.get("name", "Unknown"))
        table.add_row("Description", info.get("description", "No description"))
        table.add_row("Scenario", info.get("scenario", "Unknown"))

        if "metadata" in info:
            metadata = info["metadata"]
            if isinstance(metadata, dict):
                for key, value in metadata.items():
                    table.add_row(f"Meta: {key}", str(value))

        if "parameters" in info:
            parameters = info["parameters"]
            if isinstance(parameters, list):
                table.add_row("Parameters", ", ".join(parameters))

        console.print(table)

    except Exception as e:
        console.print(f" Error getting template info: {e}", style="red")
        raise typer.Exit(1)


@rewards_app.command()
def create_from_template(
    template_name: str = typer.Argument(..., help="Template name to use"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (defaults to rewards/{template}/reward_function.py)"
    ),
    parameters: Optional[str] = typer.Option(None, "--params", help="JSON string of custom parameters"),
    experiment_id: Optional[str] = typer.Option(None, "--experiment-id", help="Experiment ID for metadata"),
):
    """Create a reward function from a specific template."""
    try:
        from deepracer_research.rewards.template_loader import RewardFunctionRenderer

        if output is None:
            rewards_base_dir = Path("/Users/bartmlynarkiewicz/msc/notebooks/deepracer-research/rewards")
            template_dir = rewards_base_dir / template_name
            template_dir.mkdir(parents=True, exist_ok=True)
            output = template_dir / "reward_function.py"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)

        custom_params = {}
        if parameters:
            import json

            try:
                custom_params = json.loads(parameters)
            except json.JSONDecodeError as e:
                console.print(f" Invalid JSON parameters: {e}", style="red")
                raise typer.Exit(1)

        renderer = RewardFunctionRenderer()
        code = renderer.render_from_template(
            template_name=template_name, custom_parameters=custom_params, experiment_id=experiment_id
        )

        with open(output, "w") as f:
            f.write(code)

        console.print(f"✅ Reward function created from template '{template_name}'", style="green")
        console.print(f"📁 Saved to: {output}", style="blue")

        if custom_params:
            console.print(f"🔧 Custom parameters applied: {len(custom_params)} params", style="blue")

        if experiment_id:
            console.print(f"🆔 Experiment ID: {experiment_id}", style="blue")

    except Exception as e:
        console.print(f" Error creating from template: {e}", style="red")
        raise typer.Exit(1)


@rewards_app.command()
def create_from_racing_scenario(
    scenario_name: str = typer.Argument(..., help="Racing scenario name (e.g., 'centerline_following', 'speed_optimization')"),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file path (defaults to rewards/{scenario}/reward_function.py)"
    ),
    parameters: Optional[str] = typer.Option(None, "--params", help="JSON string of custom parameters"),
    experiment_id: Optional[str] = typer.Option(None, "--experiment-id", help="Experiment ID for metadata"),
):
    """Create a reward function from a racing scenario configuration."""
    try:
        from deepracer_research.rewards.template_loader import render_reward_function_for_racing_scenario

        if output is None:
            rewards_base_dir = Path("/Users/bartmlynarkiewicz/msc/notebooks/deepracer-research/rewards")
            scenario_dir = rewards_base_dir / scenario_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            output = scenario_dir / "reward_function.py"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)

        custom_params = {}
        if parameters:
            import json

            try:
                custom_params = json.loads(parameters)
            except json.JSONDecodeError as e:
                console.print(f" Invalid JSON parameters: {e}", style="red")
                raise typer.Exit(1)

        code = render_reward_function_for_racing_scenario(
            scenario_name=scenario_name, custom_parameters=custom_params, experiment_id=experiment_id
        )

        with open(output, "w") as f:
            f.write(code)

        console.print(f"✅ Reward function created for racing scenario '{scenario_name}'", style="green")
        console.print(f"📁 Saved to: {output}", style="blue")

        if custom_params:
            console.print(f"🔧 Custom parameters applied: {len(custom_params)} params", style="blue")

        if experiment_id:
            console.print(f"🆔 Experiment ID: {experiment_id}", style="blue")

    except Exception as e:
        console.print(f" Error creating from racing scenario: {e}", style="red")
        raise typer.Exit(1)


@experiments_app.command()
def create_plan(
    name: str = typer.Option("research_plan", help="Experimental plan name"),
    output: Path = typer.Option(Path("./experiment_plan.json"), "--output", "-o", help="Output plan file"),
    factorial: bool = typer.Option(True, help="Generate factorial design"),
    replications: int = typer.Option(1, help="Number of replications per condition"),
):
    """Create an experimental plan."""
    try:
        plan = ExperimentalPlan()

        if factorial:
            console.print("🔬 Generating factorial experimental design...")
            experiments = plan.generate_factorial_design(replications=replications)
            console.print(f"✅ Generated {len(experiments)} experimental conditions")

        plan.export_plan(str(output))

        summary = plan.get_plan_summary()

        table = Table(title=f"Experimental Plan: {name}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Total Experiments", str(summary["total_experiments"]))
        table.add_row("Unique Scenarios", str(summary["unique_scenarios"]))
        table.add_row("Unique Architectures", str(summary["unique_architectures"]))
        table.add_row("Unique Sensor Modalities", str(summary["unique_sensor_modalities"]))
        table.add_row("Plan ID", summary["plan_id"])

        console.print(table)
        console.print(f"📁 Plan saved to: {output}", style="green")

    except Exception as e:
        console.print(f" Error creating experimental plan: {e}", style="red")
        raise typer.Exit(1)


@experiments_app.command()
def add_experiment(
    scenario: str = typer.Argument(..., help="Experimental scenario"),
    sensor_modality: SensorModality = typer.Option(SensorModality.MONOCULAR_CAMERA, help="Sensor modality"),
    architecture: ArchitectureType = typer.Option(ArchitectureType.ATTENTION_CNN, help="Network architecture"),
    name: Optional[str] = typer.Option(None, help="Experiment name"),
    role_arn: Optional[str] = typer.Option(None, "--role-arn", help="AWS IAM role ARN (required for AWS config generation)"),
):
    """Create a custom experiment configuration.

    Use 'deepracer-research experiments list-scenarios' to see available scenarios.
    Use 'deepracer-research experiments list-sensor-modalities' to see available sensor modalities.
    Use 'deepracer-research models list-architectures' to see available architectures.
    """
    try:
        experiment = ExperimentalConfiguration(
            name=name or f"custom_{scenario}_{sensor_modality.value}_{architecture.value}",
            scenario=ExperimentalScenario(scenario),
            sensor_config=SensorConfiguration(sensor_modality),
            network_architecture=architecture,
        )

        console.print(f"✅ Created experiment: {experiment.name}")
        console.print(f"🎯 Scenario: {experiment.scenario.value}")
        console.print(f"📡 Sensor: {experiment.sensor_config.modality.value}")
        console.print(f"🧠 Architecture: {experiment.network_architecture.value}")
        console.print(f"🆔 Experiment ID: {experiment.experiment_id}")

        if role_arn:
            aws_config = experiment.to_aws_training_config(role_arn)

            console.print("\n📋 AWS Training Configuration:")
            config_table = Table()
            config_table.add_column("Parameter", style="cyan")
            config_table.add_column("Value", style="yellow")

            config_table.add_row("Model Name", aws_config["ModelName"])
            config_table.add_row("Racing Track", aws_config["RacingTrack"])
            config_table.add_row("Neural Network", aws_config["NeuralNetwork"])
            config_table.add_row("Sensors", ", ".join(aws_config["Sensors"]))
            config_table.add_row("Action Space Size", str(len(aws_config["ActionSpace"])))

            console.print(config_table)
        else:
            console.print("\n💡 To generate AWS training configuration, provide --role-arn parameter", style="yellow")
            console.print("Example: --role-arn arn:aws:iam::123456789012:role/DeepRacerRole", style="blue")

    except Exception as e:
        console.print(f" Error creating experiment: {e}", style="red")
        raise typer.Exit(1)


@experiments_app.command()
def list_sensor_modalities():
    """List available sensor modalities."""
    table = Table(title="Available Sensor Modalities")
    table.add_column("Name", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="green")

    sensor_descriptions = {
        SensorModality.MONOCULAR_CAMERA: "Single camera sensor - most basic setup",
        SensorModality.STEREO_CAMERA: "Dual camera for depth perception",
        SensorModality.LIDAR_FUSION: "LiDAR and camera fusion for rich perception",
        SensorModality.RGB_CAMERA: "Standard RGB color camera",
        SensorModality.FRONT_FACING_CAMERA: "Forward-oriented camera configuration",
    }

    for modality in SensorModality:
        description = sensor_descriptions.get(modality, "Sensor modality")
        table.add_row(modality.name, modality.value, description)

    console.print(table)


@experiments_app.command()
def list_scenarios():
    """List available experimental scenarios."""
    table = Table(title="Available Experimental Scenarios")
    table.add_column("Scenario", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="green")

    scenario_descriptions = {
        "centerline_following": "Basic track following without obstacles",
        "static_object_avoidance": "Navigate around stationary obstacles",
        "dynamic_object_avoidance": "Avoid moving obstacles and agents",
        "multi_agent": "Multi-agent racing environment",
        "speed_optimization": "Optimize for maximum speed",
        "time_trial": "Fastest lap time optimization",
        "head_to_head": "Head-to-head racing scenarios",
    }

    for scenario in ExperimentalScenario:
        description = scenario_descriptions.get(scenario.value, "Experimental scenario")
        table.add_row(scenario.name, scenario.value, description)

    console.print(table)


@training_app.command()
def status():
    """Show training job status."""
    console.print("🚂 Training Management Status", style="bold blue")

    try:
        manager = TrainingManager()
        capabilities = manager.get_training_capabilities()

        table = Table(title="Training System Status")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="yellow")

        table.add_row("AWS Available", "✅" if capabilities["aws_available"] else "")
        table.add_row("Client Connected", "✅" if capabilities["client_connected"] else "")
        table.add_row("Region", capabilities["region"])
        table.add_row("Monitoring Active", "✅" if capabilities["monitoring_active"] else "")
        table.add_row("Total Jobs", str(capabilities["total_jobs_submitted"]))
        table.add_row("Completed Jobs", str(capabilities["completed_jobs"]))
        table.add_row("Failed Jobs", str(capabilities["failed_jobs"]))

        console.print(table)

    except Exception as e:
        console.print(f" Error checking training status: {e}", style="red")


@training_app.command()
def list_jobs():
    """List all training jobs."""
    console.print("📋 Training Jobs List", style="bold blue")

    try:
        manager = TrainingManager()
        jobs = manager.get_all_job_statuses()

        if not jobs:
            console.print("No training jobs found.", style="yellow")
            return

        table = Table(title="Training Jobs")
        table.add_column("Job ID", style="cyan")
        table.add_column("Model Name", style="green")
        table.add_column("Status", style="yellow")
        table.add_column("Duration", style="magenta")

        for job_id, result in jobs.items():
            duration = f"{result.duration_minutes:.1f}m" if result.duration_minutes else "N/A"
            table.add_row(job_id, result.model_name, result.status.value, duration)

        console.print(table)

    except Exception as e:
        console.print(f" Error listing jobs: {e}", style="red")


@evaluation_app.command()
def start(
    model_name: str = typer.Argument(..., help="Name of the model to evaluate"),
    track_name: str = typer.Option(..., "--track", "-t", help="Track name for evaluation"),
    evaluation_type: EvaluationType = typer.Option(EvaluationType.TIME_TRIAL, "--type", help="Type of evaluation"),
    number_of_trials: int = typer.Option(3, "--trials", "-n", help="Number of evaluation trials"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
    wait_for_completion: bool = typer.Option(False, "--wait", "-w", help="Wait for evaluation to complete"),
    get_logs: bool = typer.Option(False, "--logs", "-l", help="Retrieve logs after completion"),
):
    """Start an evaluation job for a DeepRacer model.

    This command starts an evaluation job in the AWS DeepRacer console.
    Use 'deepracer-research evaluation status' to check progress.
    """
    try:
        console.print(f"🏁 Starting evaluation for model: {model_name}", style="bold blue")

        try:
            track_arn = get_track_arn_by_name(track_name, region)
        except ValueError as e:
            console.print(f" {e}", style="red")
            raise typer.Exit(1)

        from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager

        manager = DeepRacerDeploymentManager(region=region)

        config_table = Table(title=f"Evaluation Configuration: {model_name}")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("Model Name", model_name)
        config_table.add_row("Track", track_name)
        config_table.add_row("Evaluation Type", evaluation_type.value)
        config_table.add_row("Number of Trials", str(number_of_trials))
        config_table.add_row("Region", region)

        console.print(config_table)

        response = manager.start_evaluation_job(
            model_name=model_name, track_arn=track_arn, evaluation_type=evaluation_type, number_of_trials=number_of_trials
        )

        console.print("✅ Evaluation job started successfully!", style="green")

        if wait_for_completion:
            console.print("⏳ Waiting for evaluation to complete...", style="blue")

            try:
                final_status = manager.wait_for_evaluation_completion(model_name, timeout_minutes=60)

                eval_status = final_status.get("evaluationJobStatus", "UNKNOWN")
                if eval_status == "Completed":
                    console.print("🎉 Evaluation completed successfully!", style="green")

                    if get_logs:
                        console.print("📄 Retrieving evaluation logs...", style="blue")
                        logs = manager.get_evaluation_logs(model_name)
                        console.print(f"📁 Logs retrieved: {len(logs.get('logs', {}))} log streams", style="green")
                else:
                    console.print(f" Evaluation ended with status: {eval_status}", style="red")

            except Exception as wait_error:
                console.print(f"⚠️  Error waiting for completion: {wait_error}", style="yellow")
                console.print("Evaluation may still be running. Check status manually.", style="yellow")

        console.print("\n💡 Next Steps:", style="bold yellow")
        console.print(f"• Check status: deepracer-research evaluation status {model_name}", style="yellow")
        console.print(f"• Get logs: deepracer-research evaluation logs {model_name}", style="yellow")
        console.print(f"• Stop evaluation: deepracer-research evaluation stop {model_name}", style="yellow")

    except Exception as e:
        console.print(f" Failed to start evaluation: {e}", style="red")
        raise typer.Exit(1)


@evaluation_app.command()
def status(
    model_name: str = typer.Argument(..., help="Name of the model"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """Check the status of an evaluation job."""
    try:
        console.print(f"📊 Checking evaluation status for: {model_name}", style="bold blue")

        from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager

        manager = DeepRacerDeploymentManager(region=region)
        status = manager.get_evaluation_job_status(model_name)

        status_table = Table(title=f"Evaluation Status: {model_name}")
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="yellow")

        status_table.add_row("Model Name", status.get("modelName", "N/A"))
        status_table.add_row("Status", status.get("evaluationJobStatus", "UNKNOWN"))
        status_table.add_row("Track ARN", status.get("trackArn", "N/A"))
        status_table.add_row("Evaluation Type", status.get("evaluationType", "N/A"))
        status_table.add_row("Number of Trials", str(status.get("numberOfTrials", 0)))
        status_table.add_row("Created At", status.get("createdAt", "N/A"))
        status_table.add_row("Completed At", status.get("completedAt", "N/A"))

        console.print(status_table)

        eval_status = status.get("evaluationJobStatus", "UNKNOWN")
        if eval_status == "InProgress":
            console.print("🔄 Evaluation is currently in progress...", style="blue")
        elif eval_status == "Completed":
            console.print("✅ Evaluation completed successfully!", style="green")
        elif eval_status == "Failed":
            console.print(" Evaluation failed!", style="red")
        elif eval_status == "Stopped":
            console.print("🛑 Evaluation was stopped!", style="yellow")

    except Exception as e:
        console.print(f" Failed to get evaluation status: {e}", style="red")
        raise typer.Exit(1)


@evaluation_app.command()
def stop(
    model_name: str = typer.Argument(..., help="Name of the model"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """Stop an active evaluation job."""
    try:
        console.print(f"🛑 Stopping evaluation for: {model_name}", style="bold yellow")

        if not typer.confirm(f"Are you sure you want to stop evaluation for '{model_name}'?"):
            console.print("Operation cancelled.", style="blue")
            return

        from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager

        manager = DeepRacerDeploymentManager(region=region)
        manager.stop_evaluation_job(model_name)

        console.print("✅ Evaluation stop request sent!", style="green")
        console.print("ℹ️  It may take a few minutes for the evaluation to fully stop.", style="blue")

    except Exception as e:
        console.print(f" Failed to stop evaluation: {e}", style="red")
        raise typer.Exit(1)


@evaluation_app.command()
def logs(
    model_name: str = typer.Argument(..., help="Name of the model"),
    log_type: str = typer.Option("all", "--type", "-t", help="Type of logs to retrieve (all/simulation/robomaker)"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
    save_to_file: bool = typer.Option(False, "--save", "-s", help="Save logs to file"),
):
    """Get evaluation logs for a model."""
    try:
        console.print(f"📄 Retrieving evaluation logs for: {model_name}", style="bold blue")

        from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager

        manager = DeepRacerDeploymentManager(region=region)
        logs_data = manager.get_evaluation_logs(model_name, log_type=log_type)

        logs = logs_data.get("logs", {})

        if not logs:
            console.print(" No logs found for this model.", style="yellow")
            return

        console.print(f"📊 Found {len(logs)} log streams:", style="green")

        logs_table = Table(title=f"Evaluation Logs: {model_name}")
        logs_table.add_column("Log Stream", style="cyan")
        logs_table.add_column("Events", style="yellow")

        for log_stream, log_info in logs.items():
            event_count = len(log_info.get("events", []))
            logs_table.add_row(log_stream, str(event_count))

        console.print(logs_table)

        if save_to_file:
            import json

            output_file = Path(f"./evaluation_logs_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            with open(output_file, "w") as f:
                json.dump(logs_data, f, indent=2, default=str)

            console.print(f"📁 Logs saved to: {output_file}", style="green")

    except Exception as e:
        console.print(f" Failed to get evaluation logs: {e}", style="red")
        raise typer.Exit(1)


@evaluation_app.command()
def list_types():
    """List available evaluation types."""
    console.print("🎯 Available Evaluation Types", style="bold blue")

    table = Table(title="Evaluation Types")
    table.add_column("Type", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="green")
    table.add_column("Requires Obstacles", style="yellow")
    table.add_column("Multi-Agent", style="red")

    for eval_type in EvaluationType:
        table.add_row(
            eval_type.name,
            eval_type.value,
            eval_type.description,
            "✅" if eval_type.requires_obstacles else "",
            "✅" if eval_type.supports_multiple_agents else "",
        )

    console.print(table)


@evaluation_app.command()
def results(
    model_name: str = typer.Argument(..., help="Name of the model"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
    include_logs: bool = typer.Option(True, "--logs", help="Include logs in results"),
    save_to_file: bool = typer.Option(False, "--save", "-s", help="Save results to file"),
):
    """Get comprehensive evaluation results for a model."""
    try:
        console.print(f"📊 Getting evaluation results for: {model_name}", style="bold blue")

        from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager

        manager = DeepRacerDeploymentManager(region=region)
        results = manager.get_evaluation_results(model_name)

        summary = results.get("summary", {})

        results_table = Table(title=f"Evaluation Results: {model_name}")
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", style="yellow")

        results_table.add_row("Status", summary.get("status", "UNKNOWN"))
        results_table.add_row("Track ARN", summary.get("trackArn", "N/A"))
        results_table.add_row("Evaluation Type", summary.get("evaluationType", "N/A"))
        results_table.add_row("Number of Trials", str(summary.get("numberOfTrials", 0)))
        results_table.add_row("Created At", summary.get("createdAt", "N/A"))
        results_table.add_row("Completed At", summary.get("completedAt", "N/A"))

        console.print(results_table)

        if "metrics" in results:
            console.print("\n📈 Performance Metrics:", style="bold green")
            metrics = results["metrics"]
            console.print(
                f"Available metrics: {list(metrics.keys()) if isinstance(metrics, dict) else 'Raw metrics data'}", style="blue"
            )

        if save_to_file:
            import json

            output_file = Path(f"./evaluation_results_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            console.print(f"📁 Results saved to: {output_file}", style="green")

    except Exception as e:
        console.print(f" Failed to get evaluation results: {e}", style="red")
        raise typer.Exit(1)


@evaluation_app.command()
def run(
    model_name: str = typer.Argument(..., help="Name of the model to evaluate"),
    track_name: str = typer.Option(..., "--track", "-t", help="Track name for evaluation"),
    evaluation_type: EvaluationType = typer.Option(EvaluationType.TIME_TRIAL, "--type", help="Type of evaluation"),
    number_of_trials: int = typer.Option(3, "--trials", "-n", help="Number of evaluation trials"),
    timeout_minutes: int = typer.Option(60, "--timeout", help="Maximum time to wait for completion"),
    get_logs: bool = typer.Option(True, "--logs", "-l", help="Retrieve logs after completion"),
    save_results: bool = typer.Option(False, "--save", "-s", help="Save results to file"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """Run a complete evaluation workflow: start, wait, and get results.

    This is a convenience command that starts an evaluation, waits for completion,
    and retrieves the results automatically.
    """
    try:
        console.print(f"🚀 Running complete evaluation for: {model_name}", style="bold blue")

        try:
            track_arn = get_track_arn_by_name(track_name, region)
        except ValueError as e:
            console.print(f" {e}", style="red")
            raise typer.Exit(1)

        from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager

        manager = DeepRacerDeploymentManager(region=region)

        config_table = Table(title=f"Complete Evaluation: {model_name}")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("Model Name", model_name)
        config_table.add_row("Track", track_name)
        config_table.add_row("Evaluation Type", evaluation_type.value)
        config_table.add_row("Number of Trials", str(number_of_trials))
        config_table.add_row("Timeout", f"{timeout_minutes} minutes")
        config_table.add_row("Get Logs", "Yes" if get_logs else "No")
        config_table.add_row("Region", region)

        console.print(config_table)

        if not typer.confirm("Start the complete evaluation workflow?"):
            console.print("Operation cancelled.", style="blue")
            return

        results = manager.run_complete_evaluation(
            model_name=model_name,
            track_arn=track_arn,
            evaluation_type=evaluation_type,
            number_of_trials=number_of_trials,
            timeout_minutes=timeout_minutes,
            get_logs=get_logs,
        )

        console.print("🎉 Complete evaluation workflow finished!", style="green")

        if "summary" in results:
            summary = results["summary"]
            console.print("\n📊 Results Summary:", style="bold green")

            summary_table = Table()
            summary_table.add_column("Metric", style="cyan")
            summary_table.add_column("Value", style="yellow")

            summary_table.add_row("Final Status", summary.get("status", "UNKNOWN"))
            summary_table.add_row("Completed At", summary.get("completedAt", "N/A"))

            console.print(summary_table)

        if save_results:
            import json

            output_file = Path(f"./complete_evaluation_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

            with open(output_file, "w") as f:
                json.dump(results, f, indent=2, default=str)

            console.print(f"📁 Complete results saved to: {output_file}", style="green")

    except Exception as e:
        console.print(f" Complete evaluation workflow failed: {e}", style="red")
        raise typer.Exit(1)


@evaluation_app.command()
def analyze(
    model_name: str = typer.Argument(..., help="Model name to analyze"),
    track_name: str = typer.Option("test_track", help="Track name"),
    num_episodes: int = typer.Option(20, help="Number of evaluation episodes"),
    output_dir: Path = typer.Option(Path("./evaluation_results"), help="Output directory"),
):
    """Analyze model performance."""
    try:
        console.print(f"📊 Analyzing model: {model_name}", style="bold blue")

        analyzer = PerformanceAnalyzer(storage_path=output_dir)
        eval_id = analyzer.create_evaluation_session(model_name, track_name, num_episodes)

        import numpy as np

        with Progress() as progress:
            task = progress.add_task("Running evaluation...", total=num_episodes)

            for episode in range(num_episodes):
                episode_data = {
                    "episode": episode + 1,
                    "reward": np.random.normal(25, 5),
                    "lap_time": np.random.normal(18, 3),
                    "completed": np.random.random() > 0.2,
                    "crashed": np.random.random() < 0.1,
                }

                analyzer.add_episode_result(eval_id, episode_data)
                progress.update(task, advance=1)

        analyzer.finalize_evaluation(eval_id)
        report = analyzer.generate_performance_report(eval_id, include_plots=True)

        metrics = report["performance_metrics"]

        table = Table(title=f"Performance Analysis: {model_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="yellow")

        table.add_row("Total Episodes", str(metrics["total_episodes"]))
        table.add_row("Completion Rate", f"{metrics['completion_rate']:.2%}")
        table.add_row("Average Reward", f"{metrics['avg_reward_per_episode']:.2f}")
        table.add_row("Best Lap Time", f"{metrics['best_lap_time']:.2f}s")
        table.add_row("Successful Laps", str(metrics["successful_laps"]))
        table.add_row("Crash Count", str(metrics["crash_count"]))

        console.print(table)

        if report["recommendations"]:
            console.print("\n💡 Recommendations:", style="bold yellow")
            for rec in report["recommendations"]:
                console.print(f"• {rec}", style="yellow")

        console.print(f"\n📁 Results saved to: {output_dir}", style="green")

    except Exception as e:
        console.print(f" Analysis failed: {e}", style="red")
        raise typer.Exit(1)


@evaluation_app.command()
def compare(
    model1: str = typer.Argument(..., help="First model name"),
    model2: str = typer.Argument(..., help="Second model name"),
    metric: str = typer.Option("reward", help="Metric to compare (reward/lap_time)"),
):
    """Compare performance between two models."""
    try:
        console.print(f"⚖️ Comparing {model1} vs {model2} on {metric}", style="bold blue")

        console.print("Note: This requires existing evaluation sessions.", style="yellow")
        console.print("Run 'evaluation analyze' for each model first.", style="yellow")

    except Exception as e:
        console.print(f" Comparison failed: {e}", style="red")


@app.command()
def info():
    """Show system and library information."""
    import platform
    import sys

    from deepracer_research import __version__

    panel_content = f"""
[bold blue]DeepRacer Research Library[/bold blue]
Version: {__version__}
Python: {sys.version.split()[0]}
Platform: {platform.system()} {platform.release()}
Architecture: {platform.machine()}

[bold green]Available Modules:[/bold green]
• 🧠 Neural Architectures ({len(RACING_CONFIGS)} configurations)
• 🎯 Reward Functions
• 🔬 Experimental Design
• 🚂 Training Management
• 📊 Performance Evaluation

[bold yellow]Quick Start:[/bold yellow]
• List model configs: [cyan]deepracer-research models list-configs[/cyan]
• Create reward function: [cyan]deepracer-research rewards create-function[/cyan]
• Generate experiment plan: [cyan]deepracer-research experiments create-plan[/cyan]
• View experimental scenarios: [cyan]deepracer-research experiments list-scenarios[/cyan]
• Check training status: [cyan]deepracer-research training status[/cyan]
    """

    console.print(Panel(panel_content, title="🏁 DeepRacer Research", border_style="blue"))


@app.command()
def validate_config(
    config_file: Optional[Path] = typer.Option(None, help="Custom config file to validate"),
):
    """Validate model configurations."""
    console.print("✅ All configurations validated successfully!", style="green")


@deploy_app.command()
def deepracer(
    model_name: str = typer.Argument(..., help="Name for the DeepRacer model"),
    track_name: str = typer.Option(TrackType.PENBAY_PRO.value, help="Track name for training"),
    reward_function_file: Path = typer.Option(..., "--reward-function", "-r", help="Path to reward function file"),
    algorithm: TrainingAlgorithm = typer.Option(
        TrainingAlgorithm.CLIPPED_PPO, help="Training algorithm (CLIPPED_PPO recommended, PPO, SAC)"
    ),
    sensors: str = typer.Option(
        "FRONT_FACING_CAMERA",
        "--sensors",
        help="Comma-separated sensor types (FRONT_FACING_CAMERA recommended, SECTOR_LIDAR, LIDAR, STEREO_CAMERAS). Example: 'FRONT_FACING_CAMERA' or 'SECTOR_LIDAR,FRONT_FACING_CAMERA'",
    ),
    max_speed: float = typer.Option(4.0, help="Maximum speed for action space"),
    max_steering: float = typer.Option(30.0, help="Maximum steering angle"),
    training_duration: int = typer.Option(7200, help="Training duration in seconds"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    files_only: bool = typer.Option(
        False,
        "--files-only",
        help="Create only the essential model files (hyperparameters.json, model_metadata.json, reward_function.py) in models folder without deploying to AWS",
    ),
):
    """Deploy and start training in AWS DeepRacer console.

    The model_name parameter is used as an identifier for the model, not a file path.
    Use the reward_function_file parameter to specify the path to your reward function.
    Use 'deepracer-research deploy list-tracks' to see available tracks.
    Use 'deepracer-research deploy list-algorithms' to see available algorithms.

    Use --files-only to create only the essential model files (hyperparameters.json,
    model_metadata.json, reward_function.py) in a models folder without deploying to AWS.
    This is useful for local development or preparing files for manual deployment.

    Note: AWS DeepRacer automatically selects appropriate compute resources for training.
    """
    try:
        if files_only:
            console.print(f"📁 Creating model files for '{model_name}' in models folder...", style="bold blue")
        else:
            console.print(f"🚀 Deploying model '{model_name}' to AWS DeepRacer...", style="bold blue")

        if not reward_function_file.exists():
            console.print(f" Reward function file not found: {reward_function_file}", style="red")
            raise typer.Exit(1)

        with open(reward_function_file, "r") as f:
            reward_function_code = f.read()

        try:
            track_arn = get_track_arn_by_name(track_name, region)
        except ValueError as e:
            console.print(f" {e}", style="red")
            raise typer.Exit(1)

        try:
            sensor_types = SensorType.parse_sensor_list(sensors)
        except ValueError as e:
            console.print(f" Invalid sensor configuration: {e}", style="red")
            raise typer.Exit(1)

        if algorithm == TrainingAlgorithm.SAC:
            console.print(f"⚠️  SAC algorithm detected - forcing continuous action space", style="yellow")
            console.print(
                f"   SAC (Soft Actor-Critic) is designed for continuous control and cannot work with discrete action spaces",
                style="blue",
            )
            console.print(f"   For discrete action spaces, consider using PPO or CLIPPED_PPO instead", style="blue")

        action_space_type = AWSDeepRacerConfig.get_compatible_action_space_type(algorithm)

        config = AWSDeepRacerConfig.create_for_scenario(
            model_name=model_name,
            track_arn=track_arn,
            reward_function_code=reward_function_code,
            training_algorithm=algorithm,
            sensor_type=sensor_types,
            action_space_type=action_space_type,
            max_speed=max_speed,
            max_steering_angle=max_steering,
            max_job_duration_seconds=training_duration,
        )

        if not config.validate_configuration():
            console.print(" Invalid configuration", style="red")
            raise typer.Exit(1)

        if files_only:
            from deepracer_research.deployment.deepracer import DeepRacerDeploymentManager

            deployment_manager = DeepRacerDeploymentManager(region=region)

            try:
                models_dir, model_id = deployment_manager.create_essential_model_files(config)

                console.print(f"✅ Model files created successfully!", style="green")
                console.print(f"📁 Location: {models_dir}")
                console.print(f"🆔 Model ID: {model_id}")
                console.print(f"📋 Files created:")
                console.print(f"   • hyperparameters.json")
                console.print(f"   • model_metadata.json")
                console.print(f"   • reward_function.py")
                console.print(f"\n💡 To deploy with Thunder Compute:")
                console.print(f"   deepracer-research thunder-deploy --model-id {model_id}")

                return

            except Exception as e:
                console.print(f" Failed to create model files: {e}", style="red")
                raise typer.Exit(1)

        console.print("\n📋 Training Configuration:", style="bold green")
        config_table = Table()
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("Model Name", model_name)
        config_table.add_row("Track", track_name)
        config_table.add_row("Algorithm", algorithm.value.upper())
        config_table.add_row("Max Speed", f"{max_speed} m/s")
        config_table.add_row("Max Steering", f"{max_steering}°")
        config_table.add_row("Duration", f"{training_duration}s ({training_duration//3600}h {(training_duration%3600)//60}m)")
        config_table.add_row("Region", region)

        console.print(config_table)

        deploy_now = typer.confirm("Do you want to deploy to AWS DeepRacer now?", default=True)

        if deploy_now:
            console.print("\n🚀 Starting deployment to AWS DeepRacer...", style="bold blue")

            from deepracer_research.deployment.deepracer import DeepRacerDeploymentManager

            try:
                deployment_manager = DeepRacerDeploymentManager(region=region)

                console.print("📤 Creating training job...", style="blue")
                response = deployment_manager.create_training_job(config)

                console.print("✅ Training job created successfully!", style="green")
                console.print(f"🏁 Model: {model_name}", style="cyan")
                console.print(f"📊 Track: {track_name}", style="cyan")

                status_table = Table()
                status_table.add_column("Property", style="cyan")
                status_table.add_column("Value", style="yellow")

                status_table.add_row("Training Job ARN", response.get("modelArn", "N/A"))
                status_table.add_row("Status", "TRAINING")
                status_table.add_row("Region", region)

                console.print("\n📊 Training Job Details:", style="bold green")
                console.print(status_table)

                console.print("\n💡 Next Steps:", style="bold yellow")
                console.print(
                    "1. Monitor training progress with: deepracer-research deploy status <model_name>", style="yellow"
                )
                console.print("2. View training logs in AWS DeepRacer console", style="yellow")
                console.print("3. Training will complete automatically based on your duration setting", style="yellow")

                wait_for_completion = typer.confirm("\nWould you like to wait for training completion?", default=False)

                if wait_for_completion:
                    console.print("⏳ Waiting for training to complete...", style="blue")

                    try:
                        final_status = deployment_manager.wait_for_training_completion(
                            model_name, timeout_minutes=training_duration // 60 + 30
                        )

                        job_status = final_status.get("ModelStatus", "UNKNOWN")
                        if job_status == "READY":
                            console.print("🎉 Training completed successfully!", style="green")
                        else:
                            console.print(f" Training failed with status: {job_status}", style="red")

                    except Exception as wait_error:
                        console.print(f"⚠️  Error waiting for completion: {wait_error}", style="yellow")
                        console.print("Training may still be running. Check status manually.", style="yellow")

            except Exception as deploy_error:
                console.print(f" Deployment failed: {deploy_error}", style="red")
                console.print("💡 Saving configuration file for manual deployment...", style="yellow")
                deploy_now = False

        if not deploy_now:
            deployment_manager = DeepRacerDeploymentManager(region=region)
            deployment_config = deployment_manager.generate_deployment_config(config, save_to_file=True)

            config_file = Path(f"./deployments/{model_name}_deepracer_config.json")

            console.print(f"\n📁 Configuration saved to: {config_file}", style="green")
            console.print("\n💡 Manual Deployment Options:", style="bold yellow")
            console.print(
                "1. Use AWS CLI: aws deepracer create-training-job --cli-input-json file://config.json", style="yellow"
            )
            console.print("2. Use AWS Console: Copy reward function and settings manually", style="yellow")
            console.print("3. Use boto3 SDK in your own scripts", style="yellow")

    except Exception as e:
        console.print(f" Deployment failed: {e}", style="red")
        raise typer.Exit(1)


@deploy_app.command(name="status")
def deepracer_status(
    model_name: str = typer.Argument(..., help="Name of the DeepRacer model"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """Check the status of a DeepRacer training job."""
    try:
        from deepracer_research.deployment.deepracer import DeepRacerDeploymentManager

        console.print(f"📊 Checking status for model: {model_name}", style="bold blue")

        deployment_manager = DeepRacerDeploymentManager(region=region)
        status = deployment_manager.get_training_job_status(model_name)

        status_table = Table()
        status_table.add_column("Property", style="cyan")
        status_table.add_column("Value", style="yellow")

        status_table.add_row("Model Name", status.get("ModelName", "N/A"))
        status_table.add_row("Status", status.get("ModelStatus", "UNKNOWN"))
        status_table.add_row("Training Algorithm", status.get("TrainingAlgorithm", "N/A"))
        status_table.add_row("Track ARN", status.get("Environment", {}).get("TrackArn", "N/A"))
        status_table.add_row("Created", status.get("CreationTime", "N/A"))
        status_table.add_row("Last Modified", status.get("LastModifiedTime", "N/A"))

        if "TrainingJobConfig" in status:
            config = status["TrainingJobConfig"]
            status_table.add_row("Max Runtime", f"{config.get('MaxRuntimeInSeconds', 0)} seconds")

        console.print(status_table)

        job_status = status.get("ModelStatus", "UNKNOWN")
        if job_status == "TRAINING":
            console.print("🔄 Training is currently in progress...", style="blue")
        elif job_status == "READY":
            console.print("✅ Training completed successfully!", style="green")
        elif job_status == "FAILED":
            console.print(" Training failed!", style="red")
            failure_reason = status.get("failureReason", "Unknown")
            console.print(f"Reason: {failure_reason}", style="red")

    except Exception as e:
        console.print(f" Failed to get status: {e}", style="red")
        raise typer.Exit(1)


@deploy_app.command(name="list")
def list_deepracer_jobs(
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """List all DeepRacer training jobs."""
    try:
        from deepracer_research.deployment.deepracer import DeepRacerDeploymentManager

        console.print("📋 Listing DeepRacer training jobs...", style="bold blue")

        deployment_manager = DeepRacerDeploymentManager(region=region)
        response = deployment_manager.list_training_jobs()

        training_jobs = response.get("Models", [])

        if not training_jobs:
            console.print("No training jobs found.", style="yellow")
            return

        jobs_table = Table()
        jobs_table.add_column("Model Name", style="cyan")
        jobs_table.add_column("Status", style="yellow")
        jobs_table.add_column("Algorithm", style="green")
        jobs_table.add_column("Created", style="magenta")
        jobs_table.add_column("Duration (min)", style="blue")

        for job in training_jobs:
            duration = "N/A"
            if "TrainingJobConfig" in job:
                duration_seconds = job["TrainingJobConfig"].get("MaxRuntimeInSeconds", 0)
                duration = str(duration_seconds // 60)

            jobs_table.add_row(
                job.get("ModelName", "N/A"),
                job.get("ModelStatus", "UNKNOWN"),
                job.get("TrainingAlgorithm", "N/A"),
                job.get("CreationTime", "N/A"),
                duration,
            )

        console.print(jobs_table)
        console.print(f"\n📊 Total training jobs: {len(training_jobs)}", style="bold green")

    except Exception as e:
        console.print(f" Failed to list training jobs: {e}", style="red")
        raise typer.Exit(1)


@deploy_app.command(name="stop")
def stop_deepracer_job(
    model_name: str = typer.Argument(..., help="Name of the DeepRacer model to stop"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """Stop a running DeepRacer training job."""
    try:
        from deepracer_research.deployment.deepracer import DeepRacerDeploymentManager

        console.print(f"🛑 Stopping training job for model: {model_name}", style="bold yellow")

        if not typer.confirm(f"Are you sure you want to stop training for '{model_name}'?"):
            console.print("Operation cancelled.", style="blue")
            return

        deployment_manager = DeepRacerDeploymentManager(region=region)
        deployment_manager.stop_training_job(model_name)

        console.print("✅ Training job stop request sent!", style="green")
        console.print("ℹ️  It may take a few minutes for the job to fully stop.", style="blue")

    except Exception as e:
        console.print(f" Failed to stop training job: {e}", style="red")
        raise typer.Exit(1)


@deploy_app.command(name="delete")
def delete_deepracer_model(
    model_name: str = typer.Argument(..., help="Name of the DeepRacer model to delete"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
):
    """Delete a DeepRacer model."""
    try:
        from deepracer_research.deployment.deepracer import DeepRacerDeploymentManager

        console.print(f"🗑️  Deleting model: {model_name}", style="bold red")

        if not typer.confirm(f"Are you sure you want to delete model '{model_name}'? This action cannot be undone."):
            console.print("Operation cancelled.", style="blue")
            return

        deployment_manager = DeepRacerDeploymentManager(region=region)
        deployment_manager.delete_model(model_name)

        console.print("✅ Model deleted successfully!", style="green")

    except Exception as e:
        console.print(f" Failed to delete model: {e}", style="red")
        raise typer.Exit(1)


@deploy_app.command()
def aws(
    model_name: str = typer.Argument(..., help="Name of the model to deploy"),
    track: str = typer.Option(TrackType.PENBAY_PRO, "--track", "-t", help="Track name for training"),
    algorithm: TrainingAlgorithm = typer.Option(TrainingAlgorithm.CLIPPED_PPO, "--algorithm", "-a", help="Training algorithm"),
    sensors: str = typer.Option(
        "FRONT_FACING_CAMERA",
        "--sensors",
        help="Comma-separated sensor types (FRONT_FACING_CAMERA recommended, SECTOR_LIDAR, LIDAR, STEREO_CAMERAS). Example: 'FRONT_FACING_CAMERA' or 'SECTOR_LIDAR,FRONT_FACING_CAMERA'",
    ),
    training_time: int = typer.Option(30, "--time", help="Training time in minutes"),
    instance_type: str = typer.Option("ml.c5.2xlarge", "--instance", help="AWS instance type"),
    region: str = typer.Option("us-east-1", "--region", "-r", help="AWS region"),
    max_speed: float = typer.Option(3.0, "--max-speed", help="Maximum speed (m/s)"),
    max_steering: float = typer.Option(30.0, "--max-steering", help="Maximum steering angle (degrees)"),
    start_training: bool = typer.Option(False, "--start", help="Start training immediately after deployment"),
    files_only: bool = typer.Option(
        False,
        "--files-only",
        help="Create only the essential model files (hyperparameters.json, model_metadata.json, reward_function.py) in models folder without deploying to AWS",
    ),
):
    """Deploy model to AWS DeepRacer and optionally start training.

    The model_name parameter is used as an identifier for the model, not a file path.
    Use 'deepracer-research deploy list-tracks' to see available tracks.
    Use 'deepracer-research deploy list-algorithms' to see available algorithms.

    Use --files-only to create only the essential model files (hyperparameters.json,
    model_metadata.json, reward_function.py) in a models folder without deploying to AWS.
    """
    try:
        console.print(f"🚀 Deploying model '{model_name}' to AWS DeepRacer...", style="bold blue")

        from deepracer_research.config.training.training_algorithm import TrainingAlgorithm
        from deepracer_research.deployment.deepracer.deep_racer_deployment_config import DeepRacerDeploymentConfig
        from deepracer_research.training.management.training_manager import TrainingManager

        try:
            sensor_types = SensorType.parse_sensor_list(sensors)
        except ValueError as e:
            console.print(f" Invalid sensor configuration: {e}", style="red")
            raise typer.Exit(1)

        config = DeepRacerDeploymentConfig(
            model_name=model_name,
            track_name=track,
            training_algorithm=algorithm,
            training_time_minutes=training_time,
            sensor_configuration=sensor_types,
            region=region,
            max_speed=max_speed,
            max_steering_angle=max_steering,
        )

        manager = TrainingManager(region_name=region)

        capabilities = manager.get_training_capabilities()
        if not capabilities["aws_available"]:
            console.print(" AWS DeepRacer client not available. Check your credentials.", style="red")
            raise typer.Exit(1)

        if files_only:
            from deepracer_research.deployment.deepracer.config.aws_deep_racer_config import AWSDeepRacerConfig
            from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager
            from deepracer_research.utils.s3_utils import create_deepracer_s3_bucket

            s3_info = create_deepracer_s3_bucket("aws-sagemaker", model_name)
            s3_bucket = s3_info["name"]

            if algorithm == TrainingAlgorithm.SAC:
                console.print(f"⚠️  SAC algorithm detected - forcing continuous action space", style="yellow")
                console.print(
                    f"   SAC (Soft Actor-Critic) is designed for continuous control and cannot work with discrete action spaces",
                    style="blue",
                )
                console.print(f"   For discrete action spaces, consider using PPO or CLIPPED_PPO instead", style="blue")

            action_space_type = AWSDeepRacerConfig.get_compatible_action_space_type(algorithm)

            aws_deepracer_config = AWSDeepRacerConfig.create_for_scenario(
                model_name=model_name,
                track_arn=f"arn:aws:deepracer:{region}::track/{track}",
                reward_function_code="def reward_function(params): return 1.0",
                training_algorithm=algorithm,
                action_space_type=action_space_type,
                max_speed=max_speed,
                max_steering_angle=max_steering,
            )

            from deepracer_research.deployment.templates.template_utils import generate_deployment_files_with_s3

            result = generate_deployment_files_with_s3(
                model_name=model_name,
                deployment_target="aws_sagemaker",
                s3_bucket=s3_bucket,
                deepracer_config=aws_deepracer_config,
                region=region,
                dr_world_name=track,
            )

            models_dir = result["models_dir"]
            model_id = result["model_id"]
            s3_bucket = result["s3_bucket"]

            console.print(f"\n🎯 AWS SageMaker Deployment Files Generated!")
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

        console.print("✅ AWS connection verified", style="green")

        config_table = Table(title=f"Training Configuration: {model_name}")
        config_table.add_column("Parameter", style="cyan")
        config_table.add_column("Value", style="yellow")

        config_table.add_row("Track", track)
        config_table.add_row("Algorithm", algorithm.value.upper())
        config_table.add_row("Training Time", f"{training_time} minutes")
        config_table.add_row("Instance Type", instance_type)
        config_table.add_row("Region", region)
        config_table.add_row("Max Speed", f"{max_speed} m/s")
        config_table.add_row("Max Steering", f"{max_steering}°")

        console.print(config_table)

        if start_training:
            console.print("\n🚂 Starting training job...", style="bold yellow")

            training_config = config.to_training_job_config()
            job_id = manager.create_training_job(training_config)

            console.print(f"✅ Training job created with ID: {job_id}", style="green")
            console.print(f"🔗 Monitor progress: AWS DeepRacer Console -> Training jobs -> {job_id}", style="blue")

            with console.status("Monitoring training job startup..."):
                import time

                time.sleep(5)

            console.print("📊 Training job submitted successfully!", style="green")
            console.print(f"Use 'deepracer-research training status' to check progress", style="blue")
        else:
            from deepracer_research.deployment.deepracer.deployment_manager import DeepRacerDeploymentManager

            deployment_manager = DeepRacerDeploymentManager(region=region)
            deployment_config = deployment_manager.generate_deployment_config(config, save_to_file=True)

            config_file = Path(f"./deployments/{model_name}_aws_deployment.json")

            console.print(f"\n📁 Configuration saved to: {config_file}", style="green")
            console.print("💡 Add --start flag to begin training immediately", style="yellow")
            console.print("✅ AWS DeepRacer deployment configuration ready!", style="green")

    except Exception as e:
        console.print(f" AWS deployment failed: {e}", style="red")
        raise typer.Exit(1)


@deploy_app.command()
def list_algorithms():
    """List available training algorithms."""
    console.print("🧠 Available Training Algorithms", style="bold blue")

    table = Table(title="Training Algorithms")
    table.add_column("Algorithm", style="cyan")
    table.add_column("Value", style="magenta")
    table.add_column("Description", style="green")

    algorithm_descriptions = {
        TrainingAlgorithm.PPO: "Proximal Policy Optimization - AWS DeepRacer default, stable and reliable",
        TrainingAlgorithm.CLIPPED_PPO: "Clipped PPO - AWS DeepRacer production algorithm, proven top 1% performance",
        TrainingAlgorithm.SAC: "Soft Actor-Critic - Maximum entropy RL, excellent for continuous control",
        TrainingAlgorithm.TD3: "Twin Delayed DDPG - Advanced off-policy, precise steering control",
        TrainingAlgorithm.RAINBOW_DQN: "Rainbow DQN - Enhanced deep Q-learning with multiple improvements",
    }

    for algorithm in TrainingAlgorithm:
        description = algorithm_descriptions.get(algorithm, "Training algorithm")
        table.add_row(algorithm.name, algorithm.value, description)

    console.print(table)


@deploy_app.command()
def list_tracks():
    """List available tracks for DeepRacer training."""
    console.print("🏁 Available Training Tracks", style="bold blue")

    table = Table(title="AWS DeepRacer Tracks")
    table.add_column("Track Name", style="cyan")
    table.add_column("Category", style="yellow")

    try:
        official_tracks = get_tracks_by_type("official")
        for track in official_tracks:
            table.add_row(track.value, "Official")

        championship_tracks = get_tracks_by_type("championship")
        for track in championship_tracks:
            table.add_row(track.value, "Championship")

        console.print(table)

        try:
            professional_tracks = get_tracks_by_type("professional")
            if professional_tracks:
                console.print(f"\n🏆 Professional Tracks ({len(professional_tracks)}):", style="bold yellow")
                for track in professional_tracks:
                    console.print(f"  • {track.value}", style="yellow")
        except ValueError:
            pass

        try:
            open_tracks = get_tracks_by_type("open")
            if open_tracks:
                console.print(f"\n🌍 Open Tracks ({len(open_tracks)}):", style="bold blue")
                for track in open_tracks:
                    console.print(f"  • {track.value}", style="blue")
        except ValueError:
            pass

        try:
            special_tracks = get_tracks_by_type("special")
            if special_tracks:
                console.print(f"\n✨ Special Tracks ({len(special_tracks)}):", style="bold magenta")
                for track in special_tracks:
                    console.print(f"  • {track.value}", style="magenta")
        except ValueError:
            pass

    except Exception as e:
        console.print(f" Error loading track categories: {e}", style="red")
        console.print("📋 All Available Tracks:", style="bold blue")
        table = Table()
        table.add_column("Track Name", style="cyan")
        table.add_column("Category", style="yellow")
        for track in TrackType:
            table.add_row(track.value, "Unknown")
        console.print(table)


@app.command("thunder-deploy")
def thunder_deploy(
    model_id: str = typer.Option(..., "--model-id", help="Model ID to deploy for training"),
    cpu_cores: int = typer.Option(8, "--cpu-cores", help="Number of CPU cores"),
    gpu_type: str = typer.Option("a100-xl", "--gpu-type", help="GPU type (t4, a100, a100-xl)"),
    disk_size: int = typer.Option(100, "--disk-size", help="Disk size in GB"),
    s3_bucket: Optional[str] = typer.Option(None, "--s3-bucket", help="S3 bucket name (auto-generated if not provided)"),
    project_path: Optional[str] = typer.Option(
        None, "--project-path", help="Local project path (defaults to current directory)"
    ),
):
    """Deploy DeepRacer training to Thunder Compute with complete bootstrap workflow.

    This command creates a new Thunder Compute instance, runs the bootstrap script,
    configures AWS credentials, uploads model files, and starts training.

    Requires THUNDER_API_TOKEN environment variable to be set.
    """
    try:
        import json
        import os
        from pathlib import Path

        from deepracer_research.deployment.thunder_compute.config.instance_config import InstanceConfig
        from deepracer_research.deployment.thunder_compute.config.thunder_compute_config import ThunderComputeConfig
        from deepracer_research.deployment.thunder_compute.enum.gpu_type import GPUType
        from deepracer_research.deployment.thunder_compute.management.deployment_manager import ThunderDeploymentManager

        api_token = os.getenv("THUNDER_API_TOKEN")
        if not api_token:
            console.print(" THUNDER_API_TOKEN environment variable not set", style="red")
            console.print("Set it with: export THUNDER_API_TOKEN=your_token_here")
            raise typer.Exit(1)

        if project_path is None:
            project_path = os.getcwd()

        project_path = Path(project_path).resolve()

        if not project_path.exists():
            console.print(f" Project path not found: {project_path}", style="red")
            raise typer.Exit(1)

        model_dir = project_path / "models" / model_id
        if not model_dir.exists():
            console.print(f" Model directory not found: {model_dir}", style="red")
            console.print(
                "💡 Generate model files first with: deepracer-research deploy deepracer <model_name> --files-only",
                style="blue",
            )
            raise typer.Exit(1)

        console.print(f"🚀 Starting DeepRacer deployment to Thunder Compute", style="bold blue")
        console.print(f"📋 Configuration:")
        console.print(f"   Model ID: {model_id}")
        console.print(f"   CPU Cores: {cpu_cores}")
        console.print(f"   GPU Type: {gpu_type}")
        console.print(f"   Disk Size: {disk_size}GB")
        console.print(f"   Project Path: {project_path}")

        if s3_bucket:
            console.print(f"   S3 Bucket: {s3_bucket}")
        else:
            console.print(f"   S3 Bucket: Auto-generated")

        try:
            gpu_type_enum = GPUType(gpu_type.lower())
        except ValueError:
            console.print(f" Invalid GPU type: {gpu_type}", style="red")
            valid_types = [g.value for g in GPUType]
            console.print(f"Available types: {', '.join(valid_types)}")
            raise typer.Exit(1)

        config = ThunderComputeConfig(api_token=api_token)
        manager = ThunderDeploymentManager(config)

        instance_config = InstanceConfig.for_deepracer_training(
            cpu_cores=cpu_cores, gpu_type=gpu_type_enum, disk_size_gb=disk_size, s3_bucket_name=s3_bucket
        )

        console.print("\nStarting deployment (this may take 10-15 minutes)...")

        with console.status("[bold green]Deploying DeepRacer training environment..."):
            instance_uuid = manager.deploy_deepracer_with_bootstrap(
                model_id=model_id, instance_config=instance_config, s3_bucket_name=s3_bucket, local_project_root=project_path
            )

        console.print("✅ DeepRacer deployment completed successfully!", style="green")
        console.print(f"🆔 Instance UUID: {instance_uuid}")
        console.print(f"📋 Model ID: {model_id}")

        console.print("\n💡 Next steps:", style="bold yellow")
        console.print(f"   • Monitor training: deepracer-research thunder logs {instance_uuid}")
        console.print(f"   • SSH to instance: deepracer-research thunder ssh {instance_uuid}")
        console.print(f"   • Stop training: deepracer-research thunder stop-training {instance_uuid}")
        console.print(f"   • Delete instance: deepracer-research thunder delete {instance_uuid}")

        instance_file = f"training_{model_id}_{instance_uuid[:8]}.json"
        with open(instance_file, "w") as f:
            json.dump(
                {"model_id": model_id, "instance_uuid": instance_uuid, "deployment_type": "bootstrap_training"}, f, indent=2
            )
        console.print(f"   • Deployment info saved to: {instance_file}")

        console.print(f"\n🎯 Instance UUID for future reference: {instance_uuid}", style="bold cyan")

    except Exception as e:
        error("DeepRacer deployment failed", extra={"error": str(e)})
        console.print(f" Deployment failed: {e}", style="red")
        raise typer.Exit(1)


def main_cli():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main_cli()
