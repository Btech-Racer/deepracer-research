from typing import List

from deepracer_research.config.training.training_algorithm import TrainingAlgorithm


def get_deepracer_supported_algorithms() -> List[TrainingAlgorithm]:
    """Get algorithms officially supported by AWS DeepRacer console.

    Returns
    -------
    List[TrainingAlgorithm]
        List of algorithms supported in DeepRacer console
    """
    return TrainingAlgorithm.get_aws_supported()


def validate_algorithm_for_deepracer(algorithm: TrainingAlgorithm) -> bool:
    """Validate if an algorithm is supported by AWS DeepRacer console.

    Parameters
    ----------
    algorithm : TrainingAlgorithm
        Algorithm to validate

    Returns
    -------
    bool
        True if algorithm is supported by DeepRacer console
    """
    supported_algorithms = get_deepracer_supported_algorithms()
    return algorithm in supported_algorithms


def get_algorithm_recommendations() -> dict:
    """Get algorithm recommendations for different scenarios.

    Returns
    -------
    dict
        Dictionary mapping scenarios to recommended algorithms
    """
    return {
        "beginner": TrainingAlgorithm.PPO,
        "speed_optimization": TrainingAlgorithm.SAC,
        "stability": TrainingAlgorithm.PPO,
        "continuous_control": TrainingAlgorithm.SAC,
        "discrete_actions": TrainingAlgorithm.PPO,
    }
