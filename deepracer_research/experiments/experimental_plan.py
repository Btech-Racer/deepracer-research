import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from deepracer_research.config import ArchitectureType
from deepracer_research.experiments.config.sensor_configuration import SensorConfiguration
from deepracer_research.experiments.enums.experimental_scenario import ExperimentalScenario
from deepracer_research.experiments.enums.sensor_modality import SensorModality
from deepracer_research.experiments.experimental_configuration import ExperimentalConfiguration


class ExperimentalPlan:
    """Manages systematic experimental plan execution and factorial designs."""

    def __init__(self):
        self.experiments: List[ExperimentalConfiguration] = []
        self.results: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {
            "created_at": datetime.now(),
            "plan_id": str(uuid.uuid4())[:8],
            "status": "initialized",
        }

    def generate_factorial_design(
        self,
        scenarios: Optional[List[ExperimentalScenario]] = None,
        sensor_modalities: Optional[List[SensorModality]] = None,
        architectures: Optional[List[ArchitectureType]] = None,
        replications: int = 1,
    ) -> List[ExperimentalConfiguration]:
        """Generate complete factorial experimental design."""

        if scenarios is None:
            scenarios = [
                ExperimentalScenario.CENTERLINE_FOLLOWING,
                ExperimentalScenario.STATIC_OBJECT_AVOIDANCE,
                ExperimentalScenario.DYNAMIC_OBJECT_AVOIDANCE,
            ]

        if sensor_modalities is None:
            sensor_modalities = [SensorModality.MONOCULAR_CAMERA, SensorModality.STEREO_CAMERA, SensorModality.LIDAR_FUSION]

        if architectures is None:
            architectures = [
                ArchitectureType.ATTENTION_CNN,
                ArchitectureType.RESIDUAL_NETWORK,
                ArchitectureType.EFFICIENT_NET,
                ArchitectureType.TEMPORAL_CNN,
            ]

        experiments = []

        for scenario in scenarios:
            for sensor_modality in sensor_modalities:
                for architecture in architectures:
                    for replication in range(replications):

                        sensor_config = SensorConfiguration(modality=sensor_modality)

                        experiment = ExperimentalConfiguration(
                            name=f"exp_{scenario.value}_{sensor_modality.value}_{architecture.value}",
                            description=f"Factorial design: {scenario.value} with {sensor_modality.value} using {architecture.value}",
                            scenario=scenario,
                            sensor_config=sensor_config,
                            network_architecture=architecture,
                            tags={
                                "design_type": "factorial",
                                "replication": str(replication + 1),
                                "total_replications": str(replications),
                            },
                        )

                        experiments.append(experiment)

        self.experiments.extend(experiments)
        self.metadata["total_experiments"] = len(self.experiments)
        self.metadata["design_type"] = "factorial"

        return experiments

    def add_custom_experiment(self, experiment: ExperimentalConfiguration):
        """Add a custom experimental configuration to the plan."""
        self.experiments.append(experiment)
        self.metadata["total_experiments"] = len(self.experiments)

    def get_experiment_by_id(self, experiment_id: str) -> Optional[ExperimentalConfiguration]:
        """Retrieve experiment configuration by ID."""
        for experiment in self.experiments:
            if experiment.experiment_id == experiment_id:
                return experiment
        return None

    def get_experiments_by_scenario(self, scenario: ExperimentalScenario) -> List[ExperimentalConfiguration]:
        """Get all experiments for a specific scenario."""
        return [exp for exp in self.experiments if exp.scenario == scenario]

    def get_experiments_by_architecture(self, architecture: ArchitectureType) -> List[ExperimentalConfiguration]:
        """Get all experiments for a specific architecture."""
        return [exp for exp in self.experiments if exp.network_architecture == architecture]

    def export_plan(self, filepath: str):
        """Export experimental plan to JSON file."""
        plan_data = {
            "metadata": self.metadata,
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "name": exp.name,
                    "description": exp.description,
                    "scenario": exp.scenario.value,
                    "sensor_modality": exp.sensor_config.modality.value,
                    "network_architecture": exp.network_architecture.value,
                    "created_at": exp.created_at.isoformat(),
                    "tags": exp.tags,
                }
                for exp in self.experiments
            ],
        }

        import json

        with open(filepath, "w") as f:
            json.dump(plan_data, f, indent=2)

    def get_plan_summary(self) -> Dict[str, Any]:
        """Get summary statistics of the experimental plan."""
        if not self.experiments:
            return {"total_experiments": 0}

        scenarios = set(exp.scenario for exp in self.experiments)
        sensor_modalities = set(exp.sensor_config.modality for exp in self.experiments)
        architectures = set(exp.network_architecture for exp in self.experiments)

        return {
            "total_experiments": len(self.experiments),
            "unique_scenarios": len(scenarios),
            "unique_sensor_modalities": len(sensor_modalities),
            "unique_architectures": len(architectures),
            "scenarios": [s.value for s in scenarios],
            "sensor_modalities": [s.value for s in sensor_modalities],
            "architectures": [a.value for a in architectures],
            "plan_id": self.metadata["plan_id"],
            "created_at": self.metadata["created_at"],
            "status": self.metadata["status"],
        }
