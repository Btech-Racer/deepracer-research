import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from deepracer_research.config.aws.aws_hyperparameters import AWSHyperparameters
from deepracer_research.experiments.evaluation.analysis.statistical_analysis import StatisticalAnalysis
from deepracer_research.experiments.evaluation.results.evaluation_results import EvaluationResults
from deepracer_research.utils import info

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("husl")


class PerformanceAnalyzer:
    """Comprehensive analytics framework for DeepRacer performance evaluation."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the performance analyzer."""
        self.storage_path = storage_path or Path.cwd() / "evaluation_results"
        self.storage_path.mkdir(exist_ok=True)

        self.evaluations: Dict[str, EvaluationResults] = {}
        self.analyses: Dict[str, StatisticalAnalysis] = {}

        plt.rcParams.update(
            {
                "font.size": 12,
                "axes.titlesize": 14,
                "axes.labelsize": 12,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 10,
                "figure.dpi": 300,
                "savefig.dpi": 300,
                "savefig.bbox": "tight",
            }
        )

    def create_evaluation_session(
        self, model_name: str, track_name: str, num_episodes: int = 20, hyperparameters: Optional[AWSHyperparameters] = None
    ) -> str:
        """Create a new evaluation session.

        Parameters
        ----------
        model_name : str
            Name of the model being evaluated
        track_name : str
            Name of the track for evaluation
        num_episodes : int, optional
            Number of evaluation episodes, by default 20
        hyperparameters : Optional[AWSHyperparameters], optional
            Training hyperparameters used for the model, by default None

        Returns
        -------
        str
            Evaluation session ID
        """
        evaluation = EvaluationResults(
            model_name=model_name, track_name=track_name, num_evaluation_episodes=num_episodes, hyperparameters=hyperparameters
        )

        self.evaluations[evaluation.evaluation_id] = evaluation
        info(f"Created evaluation session {evaluation.evaluation_id} for {model_name}")

        return evaluation.evaluation_id

    def add_episode_result(self, evaluation_id: str, episode_data: Dict[str, Any]):
        """Add episode results to an evaluation session."""
        if evaluation_id not in self.evaluations:
            raise ValueError(f"Evaluation session {evaluation_id} not found")

        self.evaluations[evaluation_id].add_episode_result(episode_data)

    def finalize_evaluation(self, evaluation_id: str):
        """Finalize an evaluation session and calculate comprehensive metrics."""
        if evaluation_id not in self.evaluations:
            raise ValueError(f"Evaluation session {evaluation_id} not found")

        evaluation = self.evaluations[evaluation_id]
        evaluation.finalize_evaluation()

        self._save_evaluation_results(evaluation)

        info(f"Finalized evaluation {evaluation_id} for {evaluation.model_name}")

    def compare_models(self, evaluation_id1: str, evaluation_id2: str, metric: str = "reward") -> StatisticalAnalysis:
        """Compare performance between two models using statistical tests."""
        eval1 = self.evaluations.get(evaluation_id1)
        eval2 = self.evaluations.get(evaluation_id2)

        if not eval1 or not eval2:
            raise ValueError("One or both evaluation sessions not found")

        if metric == "reward":
            data1 = eval1.reward_history
            data2 = eval2.reward_history
        elif metric == "lap_time":
            data1 = eval1.lap_times
            data2 = eval2.lap_times
        else:
            raise ValueError(f"Unsupported metric: {metric}")

        analysis = StatisticalAnalysis(
            group1_name=eval1.model_name, group2_name=eval2.model_name, group1_data=data1, group2_data=data2
        )

        analysis.perform_t_test()

        analysis_id = f"{eval1.model_name}_vs_{eval2.model_name}_{metric}"
        self.analyses[analysis_id] = analysis

        info(f"Completed comparison: {analysis.conclusion}")
        return analysis

    def generate_performance_report(self, evaluation_id: str, include_plots: bool = True) -> Dict[str, Any]:
        """Generate comprehensive performance report for a model."""
        evaluation = self.evaluations.get(evaluation_id)
        if not evaluation:
            raise ValueError(f"Evaluation session {evaluation_id} not found")

        report = {
            "evaluation_info": {
                "evaluation_id": evaluation.evaluation_id,
                "model_name": evaluation.model_name,
                "track_name": evaluation.track_name,
                "evaluation_date": evaluation.evaluation_date.isoformat(),
                "num_episodes": evaluation.num_evaluation_episodes,
                "hyperparameters": evaluation.hyperparameters.to_dict() if evaluation.hyperparameters else None,
            },
            "performance_metrics": evaluation.metrics.to_dict(),
            "statistical_summary": self._generate_statistical_summary(evaluation),
            "recommendations": self._generate_recommendations(evaluation),
        }

        if include_plots:
            plot_paths = self._generate_performance_plots(evaluation)
            report["plot_files"] = plot_paths

        return report

    def _generate_statistical_summary(self, evaluation: EvaluationResults) -> Dict[str, Any]:
        """Generate statistical summary of evaluation results."""
        summary = {}

        if evaluation.reward_history:
            rewards = np.array(evaluation.reward_history)
            summary["reward_statistics"] = {
                "mean": float(np.mean(rewards)),
                "median": float(np.median(rewards)),
                "std": float(np.std(rewards)),
                "min": float(np.min(rewards)),
                "max": float(np.max(rewards)),
                "q1": float(np.percentile(rewards, 25)),
                "q3": float(np.percentile(rewards, 75)),
            }

        if evaluation.lap_times:
            lap_times = np.array(evaluation.lap_times)
            summary["lap_time_statistics"] = {
                "mean": float(np.mean(lap_times)),
                "median": float(np.median(lap_times)),
                "std": float(np.std(lap_times)),
                "best": float(np.min(lap_times)),
                "worst": float(np.max(lap_times)),
            }

        if evaluation.confidence_intervals:
            summary["confidence_intervals"] = {
                k: [float(v[0]), float(v[1])] for k, v in evaluation.confidence_intervals.items()
            }

        return summary

    def _generate_recommendations(self, evaluation: EvaluationResults) -> List[str]:
        """Generate performance improvement recommendations.

        Parameters
        ----------
        evaluation : EvaluationResults
            The evaluation results to analyze

        Returns
        -------
        List[str]
            List of performance improvement recommendations
        """
        recommendations = []

        if evaluation.metrics.completion_rate < 0.5:
            recommendations.append("Low completion rate suggests aggressive driving policy - consider reward function tuning")

        if evaluation.metrics.reward_std > evaluation.metrics.avg_reward_per_episode * 0.5:
            recommendations.append("High reward variance indicates inconsistent performance - extend training duration")

        if evaluation.metrics.avg_speed < 2.0:
            recommendations.append(
                "Low average speed - consider adjusting action space or reward function to encourage faster driving"
            )

        if evaluation.metrics.crash_count > evaluation.metrics.total_episodes * 0.3:
            recommendations.append("High crash rate - review obstacle avoidance and track boundary detection")

        if evaluation.hyperparameters:
            hyperparams = evaluation.hyperparameters

            if hyperparams.learning_rate > 0.001:
                if evaluation.metrics.reward_std > evaluation.metrics.avg_reward_per_episode * 0.3:
                    recommendations.append(
                        "High learning rate with unstable rewards - consider reducing learning rate for more stable training"
                    )

            elif hyperparams.learning_rate < 0.0001:
                if evaluation.metrics.episodes_to_convergence and evaluation.metrics.episodes_to_convergence > 500:
                    recommendations.append(
                        "Low learning rate may be causing slow convergence - consider increasing learning rate"
                    )

            if hyperparams.batch_size < 32:
                recommendations.append("Small batch size may cause training instability - consider increasing batch size")
            elif hyperparams.batch_size > 128:
                recommendations.append(
                    "Large batch size may slow convergence - consider reducing batch size if training is slow"
                )

            if hyperparams.entropy_coefficient < 0.005:
                if evaluation.metrics.completion_rate < 0.3:
                    recommendations.append(
                        "Low entropy coefficient may cause premature convergence - consider increasing for more exploration"
                    )
            elif hyperparams.entropy_coefficient > 0.05:
                if evaluation.metrics.reward_std > evaluation.metrics.avg_reward_per_episode * 0.4:
                    recommendations.append(
                        "High entropy coefficient may cause excessive exploration - consider reducing for more stable policy"
                    )

            if hyperparams.discount_factor < 0.95:
                recommendations.append(
                    "Low discount factor may cause short-sighted behavior - consider increasing for better long-term planning"
                )

        return recommendations

    def _generate_performance_plots(self, evaluation: EvaluationResults) -> List[str]:
        """Generate performance visualization plots."""
        plot_paths = []
        base_filename = f"{evaluation.model_name}_{evaluation.evaluation_id}"

        if evaluation.reward_history:
            plt.figure(figsize=(10, 6))
            plt.plot(evaluation.reward_history, marker="o", alpha=0.7)
            plt.title(f"Reward History - {evaluation.model_name}")
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.grid(True, alpha=0.3)

            x = np.arange(len(evaluation.reward_history))
            z = np.polyfit(x, evaluation.reward_history, 1)
            p = np.poly1d(z)
            plt.plot(x, p(x), "r--", alpha=0.8, label=f"Trend (slope: {z[0]:.2f})")
            plt.legend()

            plot_path = self.storage_path / f"{base_filename}_reward_history.png"
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(str(plot_path))

        if evaluation.lap_times:
            plt.figure(figsize=(10, 6))
            plt.hist(evaluation.lap_times, bins=20, alpha=0.7, edgecolor="black")
            plt.axvline(
                np.mean(evaluation.lap_times), color="red", linestyle="--", label=f"Mean: {np.mean(evaluation.lap_times):.2f}s"
            )
            plt.axvline(
                np.median(evaluation.lap_times),
                color="green",
                linestyle="--",
                label=f"Median: {np.median(evaluation.lap_times):.2f}s",
            )
            plt.title(f"Lap Time Distribution - {evaluation.model_name}")
            plt.xlabel("Lap Time (seconds)")
            plt.ylabel("Frequency")
            plt.legend()
            plt.grid(True, alpha=0.3)

            plot_path = self.storage_path / f"{base_filename}_lap_time_distribution.png"
            plt.savefig(plot_path)
            plt.close()
            plot_paths.append(str(plot_path))

        return plot_paths

    def _save_evaluation_results(self, evaluation: EvaluationResults):
        """Save evaluation results to JSON file."""
        filename = f"{evaluation.model_name}_{evaluation.evaluation_id}_results.json"
        filepath = self.storage_path / filename

        data = {
            "evaluation_info": {
                "evaluation_id": evaluation.evaluation_id,
                "model_name": evaluation.model_name,
                "track_name": evaluation.track_name,
                "evaluation_date": evaluation.evaluation_date.isoformat(),
                "num_episodes": evaluation.num_evaluation_episodes,
                "hyperparameters": evaluation.hyperparameters.to_dict() if evaluation.hyperparameters else None,
            },
            "metrics": evaluation.metrics.to_dict(),
            "episode_data": evaluation.episode_data,
            "statistical_summary": self._generate_statistical_summary(evaluation),
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        info(f"Saved evaluation results to {filepath}")

    def export_comparative_analysis(self, filepath: str):
        """Export all comparative analyses to JSON file."""
        analyses_data = {}

        for analysis_id, analysis in self.analyses.items():
            analyses_data[analysis_id] = {
                "test_name": analysis.test_name,
                "test_date": analysis.test_date.isoformat(),
                "significance_level": analysis.significance_level,
                "group1_name": analysis.group1_name,
                "group2_name": analysis.group2_name,
                "test_statistic": analysis.test_statistic,
                "p_value": analysis.p_value,
                "effect_size": analysis.effect_size,
                "reject_null": analysis.reject_null,
                "conclusion": analysis.conclusion,
                "group1_stats": analysis.group1_stats,
                "group2_stats": analysis.group2_stats,
            }

        with open(filepath, "w") as f:
            json.dump(analyses_data, f, indent=2)

        info(f"Exported comparative analyses to {filepath}")

    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations and analyses."""
        return {
            "total_evaluations": len(self.evaluations),
            "total_analyses": len(self.analyses),
            "models_evaluated": list(set(eval.model_name for eval in self.evaluations.values())),
            "tracks_used": list(set(eval.track_name for eval in self.evaluations.values())),
            "evaluation_ids": list(self.evaluations.keys()),
            "analysis_ids": list(self.analyses.keys()),
        }

    def analyze_hyperparameter_impact(self, metric: str = "avg_reward_per_episode") -> Dict[str, Any]:
        """Analyze the impact of hyperparameters on performance across evaluations.

        Parameters
        ----------
        metric : str, optional
            Performance metric to analyze, by default 'avg_reward_per_episode'

        Returns
        -------
        Dict[str, Any]
            Analysis results showing hyperparameter correlations with performance
        """
        evaluations_with_hyperparams = [
            eval_result for eval_result in self.evaluations.values() if eval_result.hyperparameters is not None
        ]

        if len(evaluations_with_hyperparams) < 3:
            info("Insufficient evaluations with hyperparameters for meaningful analysis")
            return {"error": "Insufficient data for hyperparameter analysis"}

        hyperparameter_data = []
        performance_values = []

        for evaluation in evaluations_with_hyperparams:
            hyperparams = evaluation.hyperparameters.to_dict()
            hyperparameter_data.append(hyperparams)

            metric_value = getattr(evaluation.metrics, metric, None)
            if metric_value is not None:
                performance_values.append(metric_value)
            else:
                hyperparameter_data.pop()

        if len(performance_values) < 3:
            return {"error": f"Insufficient data for metric '{metric}'"}

        import pandas as pd

        df = pd.DataFrame(hyperparameter_data)
        df["performance"] = performance_values

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlations = {}

        for col in numeric_cols:
            if col != "performance":
                correlation = df[col].corr(df["performance"])
                if not np.isnan(correlation):
                    correlations[col] = correlation

        best_idx = np.argmax(performance_values)
        worst_idx = np.argmin(performance_values)

        analysis_results = {
            "metric_analyzed": metric,
            "num_evaluations": len(performance_values),
            "performance_range": {
                "min": float(np.min(performance_values)),
                "max": float(np.max(performance_values)),
                "mean": float(np.mean(performance_values)),
                "std": float(np.std(performance_values)),
            },
            "hyperparameter_correlations": correlations,
            "best_configuration": {
                "performance": performance_values[best_idx],
                "hyperparameters": hyperparameter_data[best_idx],
                "model_name": evaluations_with_hyperparams[best_idx].model_name,
            },
            "worst_configuration": {
                "performance": performance_values[worst_idx],
                "hyperparameters": hyperparameter_data[worst_idx],
                "model_name": evaluations_with_hyperparams[worst_idx].model_name,
            },
            "recommendations": self._generate_hyperparameter_recommendations(correlations),
        }

        return analysis_results

    def _basic_hyperparameter_analysis(self, metric: str) -> Dict[str, Any]:
        """Basic hyperparameter analysis without sklearn dependencies.

        Parameters
        ----------
        metric : str
            Performance metric to analyze

        Returns
        -------
        Dict[str, Any]
            Basic analysis results
        """
        evaluations_with_hyperparams = [
            eval_result for eval_result in self.evaluations.values() if eval_result.hyperparameters is not None
        ]

        if not evaluations_with_hyperparams:
            return {"error": "No evaluations with hyperparameters found"}

        performance_values = []
        for evaluation in evaluations_with_hyperparams:
            metric_value = getattr(evaluation.metrics, metric, None)
            if metric_value is not None:
                performance_values.append(metric_value)

        if not performance_values:
            return {"error": f"No data available for metric '{metric}'"}

        best_idx = np.argmax(performance_values)
        best_evaluation = evaluations_with_hyperparams[best_idx]

        return {
            "metric_analyzed": metric,
            "num_evaluations": len(performance_values),
            "performance_range": {
                "min": float(np.min(performance_values)),
                "max": float(np.max(performance_values)),
                "mean": float(np.mean(performance_values)),
            },
            "best_configuration": {
                "performance": performance_values[best_idx],
                "hyperparameters": best_evaluation.hyperparameters.to_dict(),
                "model_name": best_evaluation.model_name,
            },
        }

    def _generate_hyperparameter_recommendations(self, correlations: Dict[str, float]) -> List[str]:
        """Generate recommendations based on hyperparameter correlations.

        Parameters
        ----------
        correlations : Dict[str, float]
            Correlations between hyperparameters and performance

        Returns
        -------
        List[str]
            List of hyperparameter tuning recommendations
        """
        recommendations = []

        for param, correlation in correlations.items():
            if abs(correlation) > 0.5:
                if correlation > 0:
                    recommendations.append(f"Strong positive correlation with {param} - consider increasing this parameter")
                else:
                    recommendations.append(f"Strong negative correlation with {param} - consider decreasing this parameter")
            elif abs(correlation) > 0.3:
                if correlation > 0:
                    recommendations.append(f"Moderate positive correlation with {param} - small increases may help")
                else:
                    recommendations.append(f"Moderate negative correlation with {param} - small decreases may help")

        if not recommendations:
            recommendations.append("No strong correlations found - consider exploring different hyperparameter ranges")

        return recommendations
