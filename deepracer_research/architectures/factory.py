from pathlib import Path
from typing import List, Union

import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

from deepracer_research.architectures.attention_modules import AttentionModule
from deepracer_research.architectures.residual_block import ResidualBlock
from deepracer_research.config import ArchitectureType, NetworkConfig, racing_config_manager
from deepracer_research.config.network.activation_type import ActivationType
from deepracer_research.utils.logger import info


class ModelFactory:
    """Streamlined factory for creating essential neural network architectures"""

    @staticmethod
    def create_model(config: Union[NetworkConfig, str], **overrides) -> tf.keras.Model:
        """Create a neural network model based on configuration.

        Parameters
        ----------
        config : Union[NetworkConfig, str]
            Either a NetworkConfig object or a racing config name string
        **overrides
            Optional parameter overrides when using a racing config name

        Returns
        -------
        tf.keras.Model
            The constructed neural network model

        Raises
        ------
        ValueError
            If configuration is invalid or architecture not supported
        TypeError
            If config parameter is neither NetworkConfig nor string
        KeyError
            If racing config name is not found
        """
        if isinstance(config, str):
            network_config = racing_config_manager.get_config(config, **overrides)
        elif isinstance(config, NetworkConfig):
            if overrides:
                raise ValueError("Parameter overrides can only be used with racing config names, not NetworkConfig objects")
            network_config = config
        else:
            raise TypeError(f"Config must be NetworkConfig object or string, got {type(config)}")

        if not network_config.validate():
            raise ValueError("Invalid network configuration provided")

        if network_config.architecture_type == ArchitectureType.ATTENTION_CNN:
            return ModelFactory._create_attention_cnn(network_config)
        elif network_config.architecture_type == ArchitectureType.RESIDUAL_NETWORK:
            return ModelFactory._create_residual_network(network_config)
        elif network_config.architecture_type == ArchitectureType.EFFICIENT_NET:
            return ModelFactory._create_efficient_net(network_config)
        elif network_config.architecture_type == ArchitectureType.TEMPORAL_CNN:
            return ModelFactory._create_temporal_cnn(network_config)
        elif network_config.architecture_type == ArchitectureType.LSTM_CNN:
            return ModelFactory._create_lstm_cnn(network_config)
        elif network_config.architecture_type == ArchitectureType.TRANSFORMER_VISION:
            return ModelFactory._create_transformer_vision(network_config)
        elif network_config.architecture_type == ArchitectureType.MULTI_MODAL_FUSION:
            return ModelFactory._create_multi_modal_fusion(network_config)
        elif network_config.architecture_type == ArchitectureType.NEURAL_ODE:
            return ModelFactory._create_neural_ode(network_config)
        elif network_config.architecture_type == ArchitectureType.BAYESIAN_CNN:
            return ModelFactory._create_bayesian_cnn(network_config)
        elif network_config.architecture_type == ArchitectureType.VELOCITY_AWARE:
            return ModelFactory._create_velocity_aware(network_config)
        elif network_config.architecture_type == ArchitectureType.ENSEMBLE:
            return ModelFactory._create_ensemble(network_config)
        else:
            raise ValueError(f"Unsupported architecture type: {network_config.architecture_type}")

    @staticmethod
    def create_model_with_overrides(config_name: str, **overrides) -> tf.keras.Model:
        """Create a model from racing config name with parameter overrides.

        Convenience method for creating models with parameter overrides.

        Parameters
        ----------
        config_name : str
            Name of the racing configuration
        **overrides
            Parameter overrides (e.g., learning_rate=1e-3, dropout_rate=0.2)

        Returns
        -------
        tf.keras.Model
            The constructed neural network model
        """
        return ModelFactory.create_model(config_name, **overrides)

    @staticmethod
    def _create_attention_cnn(config: NetworkConfig) -> tf.keras.Model:
        """Create CNN with spatial attention mechanisms.

        AWS DeepRacer compatible architecture with attention for focused perception.
        Excellent balance of performance and interpretability.
        """
        inputs = layers.Input(shape=config.input_shape, name="camera_input")

        x = layers.Conv2D(32, (3, 3), activation=ActivationType.RELU.value, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(64, (3, 3), activation=ActivationType.RELU.value, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = AttentionModule()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation=ActivationType.RELU.value, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = AttentionModule()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(config.output_size, activation=config.output_activation.value)(x)

        return Model(inputs, outputs, name="attention_cnn")

    @staticmethod
    def _create_residual_network(config: NetworkConfig) -> tf.keras.Model:
        """Create ResNet with skip connections for deep feature learning.

        Deep residual network enabling very deep models through skip connections.
        Proven architecture for complex visual understanding.
        """
        inputs = layers.Input(shape=config.input_shape, name="camera_input")

        x = layers.Conv2D(64, (7, 7), strides=2, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(ActivationType.RELU.value)(x)
        x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

        x = ResidualBlock(filters=64)(x)
        x = ResidualBlock(filters=64)(x)

        x = ResidualBlock(filters=128, strides=2)(x)
        x = ResidualBlock(filters=128)(x)

        x = ResidualBlock(filters=256, strides=2)(x)
        x = ResidualBlock(filters=256)(x)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(512, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(config.output_size, activation=config.output_activation.value)(x)

        return Model(inputs, outputs, name="residual_network")

    @staticmethod
    def _create_efficient_net(config: NetworkConfig) -> tf.keras.Model:
        """Create EfficientNet for optimal accuracy-efficiency trade-offs.

        Systematically scaled CNN achieving excellent performance with
        computational efficiency for real-time racing applications.
        """
        inputs = layers.Input(shape=config.input_shape, name="camera_input")

        base_model = EfficientNetB0(
            input_tensor=inputs, weights="imagenet" if config.input_shape[-1] == 3 else None, include_top=False, pooling="avg"
        )

        base_model.trainable = True
        for layer in base_model.layers[:-20]:
            layer.trainable = False

        x = base_model.output
        x = layers.Dense(256, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.2)(x)

        outputs = layers.Dense(config.output_size, activation=config.output_activation.value)(x)

        return Model(inputs, outputs, name="efficient_net")

    @staticmethod
    def _create_temporal_cnn(config: NetworkConfig) -> tf.keras.Model:
        """Create CNN with temporal modeling for motion-aware racing.

        Incorporates temporal information across multiple frames to understand
        motion dynamics and predict future states.
        """
        temporal_shape = config.input_shape
        if len(temporal_shape) == 3:
            temporal_shape = (5,) + temporal_shape

        inputs = layers.Input(shape=temporal_shape, name="temporal_input")

        x = layers.Conv3D(32, (3, 3, 3), activation=ActivationType.RELU.value, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)

        x = layers.Conv3D(64, (3, 3, 3), activation=ActivationType.RELU.value, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((1, 2, 2))(x)

        x = layers.Conv3D(128, (3, 3, 3), activation=ActivationType.RELU.value, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling3D((2, 2, 2))(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(256, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(config.output_size, activation=config.output_activation.value)(x)

        return Model(inputs, outputs, name="temporal_cnn")

    @staticmethod
    def _create_lstm_cnn(config: NetworkConfig) -> tf.keras.Model:
        """Create hybrid CNN-LSTM for sequential decision making.

        Combines convolutional feature extraction with LSTM memory for
        sequential decision making and temporal context understanding.
        """
        inputs = layers.Input(shape=config.input_shape, name="camera_input")

        x = layers.Conv2D(64, (3, 3), activation=ActivationType.RELU.value, padding="same")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(128, (3, 3), activation=ActivationType.RELU.value, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)

        x = layers.Conv2D(256, (3, 3), activation=ActivationType.RELU.value, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.RepeatVector(10)(x)

        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.LSTM(64)(x)
        x = layers.Dropout(0.3)(x)

        x = layers.Dense(128, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(config.output_size, activation=config.output_activation.value)(x)

        return Model(inputs, outputs, name="lstm_cnn")

    @staticmethod
    def _create_transformer_vision(config: NetworkConfig) -> tf.keras.Model:
        """Create Vision Transformer for global understanding.

        State-of-the-art transformer architecture providing global context
        understanding crucial for complex racing scenarios.
        """
        inputs = layers.Input(shape=config.input_shape, name="camera_input")

        patch_size = 16
        num_patches = (config.input_shape[0] // patch_size) * (config.input_shape[1] // patch_size)

        patches = layers.Conv2D(filters=256, kernel_size=patch_size, strides=patch_size, padding="valid")(inputs)
        patches = layers.Reshape((num_patches, 256))(patches)

        positions = tf.range(start=0, limit=num_patches, delta=1)
        pos_embedding = layers.Embedding(input_dim=num_patches, output_dim=256)(positions)
        encoded_patches = patches + pos_embedding

        for _ in range(4):
            attention_output = layers.MultiHeadAttention(num_heads=8, key_dim=32)(encoded_patches, encoded_patches)

            x1 = layers.Add()([attention_output, encoded_patches])
            x1 = layers.LayerNormalization(epsilon=1e-6)(x1)

            ffn_output = layers.Dense(512, activation=ActivationType.RELU.value)(x1)
            ffn_output = layers.Dense(256)(ffn_output)

            encoded_patches = layers.Add()([ffn_output, x1])
            encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)

        representation = layers.GlobalAveragePooling1D()(encoded_patches)
        x = layers.Dense(256, activation=ActivationType.RELU.value)(representation)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(128, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(config.output_size, activation=config.output_activation.value)(x)

        return Model(inputs, outputs, name="transformer_vision")

    @staticmethod
    def _create_multi_modal_fusion(config: NetworkConfig) -> tf.keras.Model:
        """Create multi-modal fusion for comprehensive sensor integration.

        Combines multiple sensor inputs (camera, telemetry) for robust
        perception and decision making in racing environments.
        """
        camera_input = layers.Input(shape=config.input_shape, name="camera_input")
        x_cam = layers.Conv2D(64, (3, 3), activation=ActivationType.RELU.value, padding="same")(camera_input)
        x_cam = layers.BatchNormalization()(x_cam)
        x_cam = layers.MaxPooling2D((2, 2))(x_cam)

        x_cam = layers.Conv2D(128, (3, 3), activation=ActivationType.RELU.value, padding="same")(x_cam)
        x_cam = layers.BatchNormalization()(x_cam)
        x_cam = layers.MaxPooling2D((2, 2))(x_cam)

        x_cam = layers.GlobalAveragePooling2D()(x_cam)
        x_cam = layers.Dense(256, activation=ActivationType.RELU.value)(x_cam)

        telemetry_input = layers.Input(shape=(10,), name="telemetry_input")
        x_tel = layers.Dense(64, activation=ActivationType.RELU.value)(telemetry_input)
        x_tel = layers.Dense(128, activation=ActivationType.RELU.value)(x_tel)

        x_cam_expanded = layers.Reshape((1, 256))(x_cam)
        x_tel_expanded = layers.Reshape((1, 128))(x_tel)

        attention_cam = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x_cam_expanded, x_tel_expanded)
        attention_tel = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x_tel_expanded, x_cam_expanded)

        x_cam_att = layers.Flatten()(attention_cam)
        x_tel_att = layers.Flatten()(attention_tel)

        fused = layers.Concatenate()([x_cam_att, x_tel_att])
        x = layers.Dense(256, activation=ActivationType.RELU.value)(fused)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation=ActivationType.RELU.value)(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(config.output_size, activation=config.output_activation.value)(x)

        return Model(inputs=[camera_input, telemetry_input], outputs=outputs, name="multi_modal_fusion")

    @staticmethod
    def _create_neural_ode(config: NetworkConfig) -> tf.keras.Model:
        """Create Neural ODE racing model with continuous dynamics.

        Advanced architecture using Neural Ordinary Differential Equations
        for continuous dynamics modeling in racing scenarios.
        """
        try:
            from deepracer_research.architectures.advanced_architectures import create_neural_ode_racing_model

            return create_neural_ode_racing_model(config)
        except ImportError:
            info("Advanced architectures not available, falling back to attention CNN")
            return ModelFactory._create_attention_cnn(config)

    @staticmethod
    def _create_bayesian_cnn(config: NetworkConfig) -> tf.keras.Model:
        """Create Bayesian CNN with uncertainty quantification.

        Advanced architecture providing uncertainty estimates for
        safety-critical racing decisions.
        """
        try:
            from deepracer_research.architectures.advanced_architectures import create_bayesian_cnn_racing_model

            return create_bayesian_cnn_racing_model(config)
        except ImportError:
            info("Advanced architectures not available, falling back to attention CNN")
            return ModelFactory._create_attention_cnn(config)

    @staticmethod
    def _create_velocity_aware(config: NetworkConfig) -> tf.keras.Model:
        """Create velocity-aware architecture with physics constraints.

        Specialized architecture incorporating physics-informed features
        for high-speed racing applications.
        """
        try:
            from deepracer_research.architectures.advanced_architectures import create_velocity_aware_racing_model

            return create_velocity_aware_racing_model(config)
        except ImportError:
            info("Advanced architectures not available, falling back to attention CNN")
            return ModelFactory._create_attention_cnn(config)

    @staticmethod
    def _create_ensemble(config: NetworkConfig) -> tf.keras.Model:
        """Create ensemble architecture combining multiple models.

        Meta-architecture that combines predictions from multiple
        specialized models for robust decision making.
        """
        try:
            from deepracer_research.architectures.advanced_architectures import create_ensemble_racing_model

            return create_ensemble_racing_model(config)
        except ImportError:
            info("Advanced architectures not available, falling back to attention CNN")
            return ModelFactory._create_attention_cnn(config)

    @classmethod
    def get_supported_architectures(cls) -> List[ArchitectureType]:
        """Get list of supported architecture types.

        Returns
        -------
        List[ArchitectureType]
            List of supported architectures
        """
        return [
            ArchitectureType.ATTENTION_CNN,
            ArchitectureType.RESIDUAL_NETWORK,
            ArchitectureType.EFFICIENT_NET,
            ArchitectureType.TEMPORAL_CNN,
            ArchitectureType.LSTM_CNN,
            ArchitectureType.TRANSFORMER_VISION,
            ArchitectureType.MULTI_MODAL_FUSION,
            ArchitectureType.NEURAL_ODE,
            ArchitectureType.BAYESIAN_CNN,
            ArchitectureType.VELOCITY_AWARE,
            ArchitectureType.ENSEMBLE,
        ]

    @classmethod
    def create_racing_model(
        cls, architecture_type: ArchitectureType, input_shape: tuple = (120, 160, 3), output_size: int = 2
    ) -> tf.keras.Model:
        """Create a model with racing-specific defaults.

        Parameters
        ----------
        architecture_type : ArchitectureType
            Type of architecture to create
        input_shape : tuple, optional
            Input shape for the model, by default (120, 160, 3)
        output_size : int, optional
            Number of output units, by default 2 (speed, steering)

        Returns
        -------
        tf.keras.Model
            The constructed racing model
        """
        from deepracer_research.config import NetworkConfig

        config = NetworkConfig(
            architecture_type=architecture_type,
            input_shape=input_shape,
            num_actions=output_size,
            output_activation=ActivationType.TANH,
        )

        return cls.create_model(config)

    @classmethod
    def create_model_with_checkpoints(
        cls, config: Union[NetworkConfig, str], output_dir: Path, model_name: str = "model", **overrides
    ) -> tf.keras.Model:
        """Create a model and save it with TensorFlow checkpoints

        Parameters
        ----------
        config : Union[NetworkConfig, str]
            Either a NetworkConfig object or a racing config name string
        output_dir : Path
            Directory to save the model and checkpoints
        model_name : str, optional
            Name for the model files, by default "model"
        **overrides
            Optional parameter overrides when using a racing config name

        Returns
        -------
        tf.keras.Model
            The created and saved model
        """
        import json
        from pathlib import Path

        model: tf.keras.Model = cls.create_model(config, **overrides)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(config, NetworkConfig):
            config_dict = config.to_dict()
            config_path = output_dir / "network_config.json"
            with open(config_path, "w") as f:
                json.dump(config_dict, f, indent=2)

        saved_model_dir = output_dir / "saved_model"
        saved_model_dir.mkdir(parents=True, exist_ok=True)
        tf.saved_model.save(model, str(saved_model_dir))

        checkpoint_dir = output_dir / "checkpoint"
        checkpoint_dir.mkdir(exist_ok=True)

        weights_path = checkpoint_dir / f"{model_name}.weights.h5"
        model.save_weights(str(weights_path))

        iteration = 0
        step = 0
        aws_checkpoint_name = f"{iteration}_Step-{step}.ckpt"
        aws_ckpt_path = checkpoint_dir / aws_checkpoint_name

        try:
            checkpoint = tf.train.Checkpoint(model=model)
            save_path = checkpoint.save(str(aws_ckpt_path))
            info(f"Created TensorFlow checkpoint at: {save_path}")

            import glob

            ckpt_pattern = str(aws_ckpt_path) + "-*"
            saved_files = glob.glob(ckpt_pattern)

            for saved_file in saved_files:
                saved_path = Path(saved_file)
                if "-1." in saved_file:
                    new_name = saved_file.replace("-1.", ".")
                    new_path = Path(new_name)
                    saved_path.rename(new_path)
                    info(f"Renamed {saved_path.name} -> {new_path.name}")

            meta_path = checkpoint_dir / f"{aws_checkpoint_name}.meta"
            ModelFactory._create_meta_file(meta_path, model)

        except Exception as e:
            info(f"Checkpoint creation failed: {e}")
            info("Model weights are available in .weights.h5 format")

        coach_checkpoint_path = checkpoint_dir / ".coach_checkpoint"
        with open(coach_checkpoint_path, "w") as f:
            f.write(f"{aws_checkpoint_name}\n")

        deepracer_checkpoints = {"last_checkpoint": aws_checkpoint_name, "checkpoints": [aws_checkpoint_name]}

        import json

        deepracer_checkpoints_path = checkpoint_dir / "deepracer_checkpoints.json"
        with open(deepracer_checkpoints_path, "w") as f:
            json.dump(deepracer_checkpoints, f, indent=2)

        return model

    @staticmethod
    def _create_meta_file(meta_path: Path, model: tf.keras.Model):
        """Create a proper .meta file for AWS DeepRacer compatibility.

        Parameters
        ----------
        meta_path : Path
            Path where to save the .meta file
        model : tf.keras.Model
            The Keras model
        """
        try:

            @tf.function
            def model_func(inputs):
                return model(inputs)

            input_shape = list(model.input_shape)
            if input_shape[0] is None:
                input_shape[0] = 1

            input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32, name="inputs")]
            concrete_func = model_func.get_concrete_function(*input_signature)

            from tensorflow.core.protobuf import meta_graph_pb2

            meta_graph_def = meta_graph_pb2.MetaGraphDef()

            meta_graph_def.graph_def.CopyFrom(concrete_func.graph.as_graph_def())

            signature_def = meta_graph_pb2.SignatureDef()
            signature_def.inputs["inputs"].CopyFrom(tf.compat.v1.utils.build_tensor_info(concrete_func.inputs[0]))
            signature_def.outputs["output_0"].CopyFrom(tf.compat.v1.utils.build_tensor_info(concrete_func.outputs[0]))
            signature_def.method_name = tf.saved_model.PREDICT_METHOD_NAME

            meta_graph_def.signature_def["serving_default"].CopyFrom(signature_def)

            with open(meta_path, "wb") as f:
                f.write(meta_graph_def.SerializeToString())

            info(f"Created proper .meta file with signatures: {meta_path.name}")

        except Exception as e:
            info(f"Could not create proper .meta file: {e}")
            try:
                from tensorflow.core.protobuf import meta_graph_pb2

                meta_graph_def = meta_graph_pb2.MetaGraphDef()

                @tf.function
                def basic_func(inputs):
                    return model(inputs)

                input_shape = list(model.input_shape)
                if input_shape[0] is None:
                    input_shape[0] = 1

                input_signature = [tf.TensorSpec(shape=input_shape, dtype=tf.float32)]
                concrete = basic_func.get_concrete_function(*input_signature)
                meta_graph_def.graph_def.CopyFrom(concrete.graph.as_graph_def())

                with open(meta_path, "wb") as f:
                    f.write(meta_graph_def.SerializeToString())
                info(f"Created basic .meta file as fallback: {meta_path.name}")

            except Exception as e2:
                info(f"All .meta file creation attempts failed: {e2}")
                with open(meta_path, "wb") as f:
                    f.write(b"")
                info(f"Created empty .meta file as last resort: {meta_path.name}")


def create_attention_cnn_model(input_shape: tuple = (120, 160, 3), output_size: int = 2) -> tf.keras.Model:
    """Create an attention CNN model with default racing parameters."""
    return ModelFactory.create_racing_model(ArchitectureType.ATTENTION_CNN, input_shape, output_size)


def create_efficient_net_model(input_shape: tuple = (120, 160, 3), output_size: int = 2) -> tf.keras.Model:
    """Create an EfficientNet model with default racing parameters."""
    return ModelFactory.create_racing_model(ArchitectureType.EFFICIENT_NET, input_shape, output_size)


def create_transformer_vision_model(input_shape: tuple = (120, 160, 3), output_size: int = 2) -> tf.keras.Model:
    """Create a Vision Transformer model with default racing parameters."""
    return ModelFactory.create_racing_model(ArchitectureType.TRANSFORMER_VISION, input_shape, output_size)


def create_deepracer_model_with_checkpoints(
    architecture_type: ArchitectureType,
    output_dir: Path,
    input_shape: tuple = (160, 120, 3),
    output_size: int = 2,
    model_name: str = "deepracer_model",
) -> tf.keras.Model:
    """Create a DeepRacer-compatible model with checkpoints.

    Parameters
    ----------
    architecture_type : ArchitectureType
        Type of neural network architecture to create
    output_dir : Path
        Directory to save the model and checkpoint files
    input_shape : tuple, optional
        Input shape matching DeepRacer camera format, by default (160, 120, 3)
    output_size : int, optional
        Number of output actions, by default 2 (speed, steering)
    model_name : str, optional
        Name for the model files, by default "deepracer_model"

    Returns
    -------
    tf.keras.Model
        The created model with checkpoints saved to output_dir
    """
    from deepracer_research.config import NetworkConfig

    config = NetworkConfig(
        architecture_type=architecture_type,
        input_shape=input_shape,
        num_actions=output_size,
        output_activation=ActivationType.TANH,
    )

    return ModelFactory.create_model_with_checkpoints(config=config, output_dir=output_dir, model_name=model_name)


def create_advanced_model_with_checkpoints(
    architecture_type: ArchitectureType,
    output_dir: Path,
    input_shape: tuple = (160, 120, 3),
    output_size: int = 2,
    model_name: str = "advanced_model",
) -> tf.keras.Model:
    """Create an advanced/experimental model with checkpoints.

    Parameters
    ----------
    architecture_type : ArchitectureType
        Type of advanced architecture to create
    output_dir : Path
        Directory to save the model and checkpoint files
    input_shape : tuple, optional
        Input shape matching DeepRacer camera format, by default (160, 120, 3)
    output_size : int, optional
        Number of output actions, by default 2 (speed, steering)
    model_name : str, optional
        Name for the model files, by default "advanced_model"

    Returns
    -------
    tf.keras.Model
        The created model with checkpoints saved to output_dir
    """
    from deepracer_research.config import NetworkConfig

    advanced_architectures = {
        ArchitectureType.NEURAL_ODE,
        ArchitectureType.BAYESIAN_CNN,
        ArchitectureType.VELOCITY_AWARE,
        ArchitectureType.ENSEMBLE,
    }

    if architecture_type not in advanced_architectures:
        info(f"{architecture_type.value} is not an advanced architecture, using standard creation")
        return create_deepracer_model_with_checkpoints(
            architecture_type=architecture_type,
            output_dir=output_dir,
            input_shape=input_shape,
            output_size=output_size,
            model_name=model_name,
        )

    try:
        config = NetworkConfig(
            architecture_type=architecture_type,
            input_shape=input_shape,
            num_actions=output_size,
            output_activation=ActivationType.TANH,
        )

        return ModelFactory.create_model_with_checkpoints(config=config, output_dir=output_dir, model_name=model_name)
    except Exception as e:
        info(f"Failed to create {architecture_type.value} model: {e}")
        info("Falling back to attention CNN with checkpoints")

        fallback_config = NetworkConfig(
            architecture_type=ArchitectureType.ATTENTION_CNN,
            input_shape=input_shape,
            num_actions=output_size,
            output_activation=ActivationType.TANH,
        )

        return ModelFactory.create_model_with_checkpoints(
            config=fallback_config, output_dir=output_dir, model_name=f"{model_name}_fallback"
        )
