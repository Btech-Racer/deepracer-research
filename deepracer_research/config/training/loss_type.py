from enum import Enum, unique


@unique
class LossType(str, Enum):
    """Essential loss functions for AWS DeepRacer and autonomous racing research"""

    MSE = "mse"
    """Mean Squared Error (MSE) loss function.

    The most basic and widely used loss function for regression tasks.
    Computes the average of the squared differences between predicted and actual values.
    """

    CATEGORICAL_CROSSENTROPY = "categorical_crossentropy"
    """Categorical Crossentropy loss function.

    Used for classification tasks where the target is a categorical variable.
    Computes the cross-entropy between the predicted and actual class distributions.
    """

    MAE = "mae"
    """Mean Absolute Error (MAE) loss function.

    Computes the average of the absolute differences between predicted and actual values.
    """

    HUBER = "huber"
    """Huber Loss (Robust Regression Loss).

    AWS DeepRacer's default loss function. Combines the best properties of MSE
    and MAE - quadratic for small errors (sensitive to small deviations) and
    linear for large errors (robust to outliers). Excellent for racing control
    where both precision and robustness are important.

    Mathematical Definition:
        L_δ(y, f(x)) = {
            0.5(y - f(x))²           if |y - f(x)| ≤ δ
            δ|y - f(x)| - 0.5δ²      otherwise
        }

    Use Cases:
    - Default AWS DeepRacer training
    - General autonomous racing applications
    - When outliers are present in training data
    - Continuous control with mixed precision needs

    Reference: Huber, P. J. (1964). Robust Estimation of a Location Parameter
    """

    FOCAL_LOSS = "focal_loss"
    """Focal Loss for Addressing Class Imbalance (Lin et al., 2017).

    Designed to address class imbalance by down-weighting easy examples and
    focusing learning on hard negatives. Crucial for racing scenarios where
    certain actions (like emergency braking) are rare but critical.

    Mathematical Definition:
        FL(p_t) = -α_t(1-p_t)^γ log(p_t)
        where p_t is the model's estimated probability for the true class

    Use Cases:
    - Imbalanced racing datasets (rare critical actions)
    - Multi-class action classification
    - Safety-critical driving behaviors
    - When certain racing maneuvers are underrepresented

    Reference: Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV
    """

    POLICY_GRADIENT_LOSS = "policy_gradient_loss"
    """Policy Gradient Loss for Reinforcement Learning.

    Core loss function for policy-based RL algorithms like PPO and SAC.
    Optimizes the policy directly by maximizing expected cumulative reward.
    Essential for autonomous racing where action selection is critical.

    Mathematical Definition:
        L_PG = -E[log π(a|s) * A(s,a)]
        where A(s,a) is the advantage function

    Use Cases:
    - PPO and SAC training (AWS DeepRacer supported)
    - Policy optimization in racing environments
    - Direct action probability optimization
    - Exploration-exploitation balance in racing

    Reference: Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms
    """

    MULTI_MODAL_CONTRASTIVE_LOSS = "multi_modal_contrastive_loss"
    """Multi-Modal Contrastive Loss for Racing Perception.

    Combines visual, sensor, and action data for robust representation learning.
    Critical for racing applications that integrate camera data with telemetry
    (speed, steering angle, GPS) for comprehensive decision making.

    Mathematical Definition:
        L_contrast = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
        where z_i, z_j are positive pairs from different modalities

    Use Cases:
    - Camera + LiDAR + telemetry fusion
    - Robust perception under varying conditions
    - Cross-modal representation learning
    - Handling sensor failures gracefully

    Reference: Chen, T., et al. (2020). A Simple Framework for Contrastive Learning
    """

    COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS = "combined_aleatoric_epistemic_uncertainty_loss"
    """Combined Aleatoric and Epistemic Uncertainty Loss (Kendall & Gal, 2017).

    Quantifies both aleatoric uncertainty (data noise) and epistemic uncertainty
    (model uncertainty). Critical for safe autonomous racing by providing
    confidence estimates and enabling risk-aware decision making.

    Mathematical Definition:
        L_total = L_aleatoric + L_epistemic
        L_aleatoric = (1/2σ²)||y - f(x)||² + (1/2)log(σ²)
        L_epistemic = KL[q(w)||p(w)] (variational inference)

    Use Cases:
    - Safety-critical racing applications
    - Risk-aware decision making
    - Model confidence estimation
    - Handling uncertainty in racing conditions
    - Research in probabilistic racing models

    Key Benefits:
    - Separates different types of uncertainty
    - Enables calibrated confidence estimation
    - Supports safe exploration in racing
    - Critical for deployment validation

    Reference: Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning?
    """

    @classmethod
    def get_aws_supported(cls) -> list:
        """Get loss functions officially supported by AWS DeepRacer."""
        return [cls.HUBER]

    @classmethod
    def get_regression_losses(cls) -> list:
        """Get loss functions suitable for regression tasks."""
        return [cls.HUBER, cls.COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS]

    @classmethod
    def get_classification_losses(cls) -> list:
        """Get loss functions suitable for classification tasks."""
        return [cls.FOCAL_LOSS]

    @classmethod
    def get_rl_losses(cls) -> list:
        """Get loss functions for reinforcement learning."""
        return [cls.POLICY_GRADIENT_LOSS]

    @classmethod
    def get_multi_modal_losses(cls) -> list:
        """Get loss functions for multi-modal learning."""
        return [cls.MULTI_MODAL_CONTRASTIVE_LOSS]

    @classmethod
    def get_uncertainty_losses(cls) -> list:
        """Get loss functions that quantify uncertainty."""
        return [cls.COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS]

    @classmethod
    def get_research_losses(cls) -> list:
        """Get advanced loss functions for research applications."""
        return [cls.FOCAL_LOSS, cls.MULTI_MODAL_CONTRASTIVE_LOSS, cls.COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS]

    def get_description(self) -> str:
        """Get concise description of the loss function."""
        descriptions = {
            self.HUBER: "AWS default - robust to outliers, balances precision and stability",
            self.FOCAL_LOSS: "Addresses class imbalance - focuses on hard examples and rare actions",
            self.POLICY_GRADIENT_LOSS: "RL policy optimization - maximizes expected rewards",
            self.MULTI_MODAL_CONTRASTIVE_LOSS: "Multi-sensor fusion - combines vision, sensors, and actions",
            self.COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS: "Uncertainty quantification - separates data and model uncertainty",
        }
        return descriptions[self]

    def get_use_case(self) -> str:
        """Get primary use case for this loss function."""
        use_cases = {
            self.HUBER: "General racing, AWS DeepRacer, robust continuous control",
            self.FOCAL_LOSS: "Imbalanced datasets, rare critical actions, safety behaviors",
            self.POLICY_GRADIENT_LOSS: "PPO/SAC training, policy optimization, RL algorithms",
            self.MULTI_MODAL_CONTRASTIVE_LOSS: "Sensor fusion, camera+telemetry, robust perception",
            self.COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS: "Safety-critical racing, confidence estimation, risk-aware decisions",
        }
        return use_cases[self]

    def get_reference(self) -> str:
        """Get reference for the loss function."""
        references = {
            self.HUBER: "Huber, P. J. (1964). Robust Estimation of a Location Parameter",
            self.FOCAL_LOSS: "Lin, T. Y., et al. (2017). Focal Loss for Dense Object Detection. ICCV",
            self.POLICY_GRADIENT_LOSS: "Williams, R. J. (1992). Simple Statistical Gradient-Following Algorithms",
            self.MULTI_MODAL_CONTRASTIVE_LOSS: "Chen, T., et al. (2020). A Simple Framework for Contrastive Learning",
            self.COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS: "Kendall, A., & Gal, Y. (2017). What Uncertainties Do We Need in Bayesian Deep Learning?",
        }
        return references[self]

    def requires_special_setup(self) -> bool:
        """Check if this loss function requires special model setup."""
        special_setup = {
            self.HUBER: False,
            self.FOCAL_LOSS: True,
            self.POLICY_GRADIENT_LOSS: True,
            self.MULTI_MODAL_CONTRASTIVE_LOSS: True,
            self.COMBINED_ALEATORIC_EPISTEMIC_UNCERTAINTY_LOSS: True,
        }
        return special_setup[self]
