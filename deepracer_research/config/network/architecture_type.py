from enum import Enum, unique


@unique
class ArchitectureType(str, Enum):
    """Essential neural architectures for AWS DeepRacer and autonomous racing research"""

    ATTENTION_CNN = "attention_cnn"
    """CNN with spatial attention mechanisms for focused perception.

    Combines convolutional feature extraction with attention mechanisms to focus
    on relevant parts of the track. Excellent balance of performance and interpretability.

    Key Features:
    - Spatial attention for track focus
    - Efficient feature extraction
    - Good baseline performance
    - AWS DeepRacer compatible

    Use Cases:
    - General racing applications
    - Track centerline following
    - Basic autonomous navigation

    Reference: Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module
    """

    RESIDUAL_NETWORK = "residual_network"
    """ResNet with skip connections for deep feature learning.

    Deep residual networks that enable training of very deep models through
    skip connections. Proven architecture for complex visual understanding.

    Key Features:
    - Skip connections prevent vanishing gradients
    - Deep feature hierarchies
    - Robust performance across scenarios
    - Well-established architecture

    Use Cases:
    - Complex track layouts
    - Multi-object detection
    - Robust visual understanding

    Reference: He, K., et al. (2016). Deep Residual Learning for Image Recognition
    """

    TEMPORAL_CNN = "temporal_cnn"
    """CNN with temporal modeling for motion-aware racing.

    Incorporates temporal information across multiple frames to understand
    motion dynamics and predict future states for better racing decisions.

    Key Features:
    - Multi-frame temporal processing
    - Motion-aware feature extraction
    - Velocity and acceleration understanding
    - Predictive capabilities

    Use Cases:
    - High-speed racing scenarios
    - Dynamic obstacle avoidance
    - Predictive path planning

    Reference: Tran, D., et al. (2015). Learning Spatiotemporal Features with 3D CNNs
    """

    TRANSFORMER_VISION = "transformer_vision"
    """Vision Transformer for global understanding.

    State-of-the-art transformer architecture adapted for visual perception.
    Provides global context understanding crucial for racing scenarios.

    Key Features:
    - Global context modeling
    - Self-attention mechanisms
    - Scalable architecture
    - Strong performance on complex scenes

    Use Cases:
    - Complex racing environments
    - Multi-object scene understanding
    - Research applications

    Reference: Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words
    """

    MULTI_MODAL_FUSION = "multi_modal_fusion"
    """Multi-modal fusion for comprehensive sensor integration.

    Combines multiple sensor inputs (camera, LiDAR, telemetry) for robust
    perception and decision making in racing environments.

    Key Features:
    - Camera + sensor data fusion
    - Robust to sensor failures
    - Comprehensive perception
    - Real-world deployment ready

    Use Cases:
    - Production racing systems
    - Safety-critical applications
    - Multi-sensor platforms

    Reference: Ramachandran, P., et al. (2019). Stand-Alone Self-Attention
    """

    EFFICIENT_NET = "efficient_net"
    """EfficientNet for optimal accuracy-efficiency trade-offs.

    Systematically scaled CNN architecture that achieves excellent performance
    with computational efficiency, crucial for real-time racing applications.

    Key Features:
    - Optimal accuracy-efficiency balance
    - Systematic scaling methodology
    - Real-time inference capability
    - Mobile/edge deployment ready

    Use Cases:
    - Real-time racing applications
    - Edge computing deployment
    - Resource-constrained environments

    Reference: Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling
    """

    LSTM_CNN = "lstm_cnn"
    """Hybrid CNN-LSTM for sequential decision making.

    Combines convolutional feature extraction with LSTM memory for sequential
    decision making and temporal context understanding in racing.

    Key Features:
    - Memory-based decision making
    - Sequential pattern recognition
    - Temporal context awareness
    - Proven hybrid architecture

    Use Cases:
    - Sequential racing maneuvers
    - Track memory and learning
    - Adaptive racing strategies

    Reference: Donahue, J., et al. (2015). Long-term Recurrent Convolutional Networks
    """

    # Advanced experimental architectures
    NEURAL_ODE = "neural_ode"
    """Neural Ordinary Differential Equation networks for continuous dynamics.

    Advanced architecture using continuous-depth neural networks to model
    smooth trajectory dynamics in racing scenarios.

    Key Features:
    - Continuous dynamics modeling
    - Smooth trajectory generation
    - Memory efficient training
    - Adaptive depth

    Use Cases:
    - Trajectory optimization
    - Smooth control generation
    - Research applications

    Reference: Chen, R., et al. (2018). Neural Ordinary Differential Equations
    """

    BAYESIAN_CNN = "bayesian_cnn"
    """Bayesian Convolutional Neural Network with uncertainty quantification.

    CNN with uncertainty estimation through Bayesian inference, providing
    confidence measures for racing decisions.

    Key Features:
    - Uncertainty quantification
    - Robust to data distribution shifts
    - Confidence-aware decisions
    - Safety-critical applications

    Use Cases:
    - Safety-critical racing
    - Uncertainty-aware control
    - Risk assessment

    Reference: Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation
    """

    VELOCITY_AWARE = "velocity_aware"
    """Velocity-aware architecture with physics-informed features.

    Specialized architecture that explicitly models velocity and physics
    constraints for racing applications.

    Key Features:
    - Physics-informed design
    - Velocity modeling
    - Dynamic constraints
    - Racing-specific features

    Use Cases:
    - High-speed racing
    - Physics-based control
    - Speed optimization
    """

    ENSEMBLE = "ensemble"
    """Ensemble architecture combining multiple specialized models.

    Meta-architecture that combines predictions from multiple specialized
    models for robust decision making.

    Key Features:
    - Multiple model combination
    - Robust predictions
    - Uncertainty reduction
    - Performance improvement

    Use Cases:
    - Production racing systems
    - High-performance applications
    - Critical decision making
    """

    def get_complexity_level(self) -> str:
        """Get complexity level for this architecture."""
        basic = {self.ATTENTION_CNN, self.EFFICIENT_NET}
        intermediate = {self.RESIDUAL_NETWORK, self.TEMPORAL_CNN, self.LSTM_CNN}
        advanced = {self.TRANSFORMER_VISION, self.MULTI_MODAL_FUSION}
        experimental = {self.NEURAL_ODE, self.BAYESIAN_CNN, self.VELOCITY_AWARE, self.ENSEMBLE}

        if self in basic:
            return "Basic"
        elif self in intermediate:
            return "Intermediate"
        elif self in advanced:
            return "Advanced"
        elif self in experimental:
            return "Experimental"
        else:
            return "Unknown"

    @classmethod
    def get_cnn_architectures(cls) -> list:
        """Get pure CNN-based architectures."""
        return [cls.ATTENTION_CNN, cls.RESIDUAL_NETWORK, cls.EFFICIENT_NET]

    @classmethod
    def get_temporal_architectures(cls) -> list:
        """Get architectures with temporal modeling."""
        return [cls.TEMPORAL_CNN, cls.LSTM_CNN]

    @classmethod
    def get_transformer_architectures(cls) -> list:
        """Get transformer-based architectures."""
        return [cls.TRANSFORMER_VISION]

    @classmethod
    def get_multi_modal_architectures(cls) -> list:
        """Get multi-modal fusion architectures."""
        return [cls.MULTI_MODAL_FUSION]

    @classmethod
    def get_aws_compatible(cls) -> list:
        """Get architectures compatible with AWS DeepRacer."""
        return [cls.ATTENTION_CNN, cls.RESIDUAL_NETWORK, cls.TEMPORAL_CNN, cls.EFFICIENT_NET, cls.LSTM_CNN]

    @classmethod
    def get_research_architectures(cls) -> list:
        """Get advanced architectures for research applications."""
        return [cls.TRANSFORMER_VISION, cls.MULTI_MODAL_FUSION, cls.TEMPORAL_CNN]

    @classmethod
    def get_real_time_architectures(cls) -> list:
        """Get architectures suitable for real-time inference."""
        return [cls.ATTENTION_CNN, cls.EFFICIENT_NET, cls.RESIDUAL_NETWORK]

    def get_description(self) -> str:
        """Get concise description of the architecture."""
        descriptions = {
            self.ATTENTION_CNN: "CNN with attention - focused track perception, AWS compatible",
            self.RESIDUAL_NETWORK: "Deep ResNet - robust feature learning with skip connections",
            self.TEMPORAL_CNN: "Motion-aware CNN - temporal modeling for dynamic racing",
            self.TRANSFORMER_VISION: "Vision Transformer - global context understanding",
            self.MULTI_MODAL_FUSION: "Multi-sensor fusion - camera + telemetry integration",
            self.EFFICIENT_NET: "Efficient CNN - optimal accuracy-speed trade-off",
            self.LSTM_CNN: "Hybrid CNN-LSTM - memory-based sequential decisions",
        }
        return descriptions[self]

    def get_use_case(self) -> str:
        """Get primary use case for this architecture."""
        use_cases = {
            self.ATTENTION_CNN: "General racing, centerline following, AWS DeepRacer",
            self.RESIDUAL_NETWORK: "Complex tracks, robust visual understanding",
            self.TEMPORAL_CNN: "High-speed racing, motion prediction, dynamic scenarios",
            self.TRANSFORMER_VISION: "Complex environments, research applications",
            self.MULTI_MODAL_FUSION: "Production systems, safety-critical applications",
            self.EFFICIENT_NET: "Real-time inference, edge deployment",
            self.LSTM_CNN: "Sequential maneuvers, track memory, adaptive strategies",
            self.NEURAL_ODE: "Trajectory optimization, smooth control generation",
            self.BAYESIAN_CNN: "Safety-critical racing, uncertainty-aware control",
            self.VELOCITY_AWARE: "High-speed racing, physics-based control",
            self.ENSEMBLE: "Production racing systems, critical decision making",
        }
        return use_cases[self]

    def get_reference(self) -> str:
        """Get reference for the architecture."""
        references = {
            self.ATTENTION_CNN: "Woo, S., et al. (2018). CBAM: Convolutional Block Attention Module",
            self.RESIDUAL_NETWORK: "He, K., et al. (2016). Deep Residual Learning for Image Recognition",
            self.TEMPORAL_CNN: "Tran, D., et al. (2015). Learning Spatiotemporal Features with 3D CNNs",
            self.TRANSFORMER_VISION: "Dosovitskiy, A., et al. (2020). An Image is Worth 16x16 Words",
            self.MULTI_MODAL_FUSION: "Ramachandran, P., et al. (2019). Stand-Alone Self-Attention",
            self.EFFICIENT_NET: "Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling",
            self.LSTM_CNN: "Donahue, J., et al. (2015). Long-term Recurrent Convolutional Networks",
            self.NEURAL_ODE: "Chen, R., et al. (2018). Neural Ordinary Differential Equations",
            self.BAYESIAN_CNN: "Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation",
            self.VELOCITY_AWARE: "Physics-informed neural networks for racing applications",
            self.ENSEMBLE: "Ensemble methods for robust prediction",
        }
        return references[self]

    def requires_temporal_input(self) -> bool:
        """Check if architecture requires temporal/sequential input."""
        temporal_architectures = {self.TEMPORAL_CNN, self.LSTM_CNN}
        return self in temporal_architectures

    def requires_multi_modal_input(self) -> bool:
        """Check if architecture requires multi-modal input."""
        return self == self.MULTI_MODAL_FUSION
