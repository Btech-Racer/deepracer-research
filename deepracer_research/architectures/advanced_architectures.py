from typing import List

import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import Model, layers

from deepracer_research.architectures.attention_modules import AttentionModule
from deepracer_research.architectures.residual_block import ResidualBlock
from deepracer_research.config.network.network_config import NetworkConfig


class NeuralODERacingNetwork(layers.Layer):
    """Neural Ordinary Differential Equation network for continuous dynamics modeling.

    Based on Chen et al. (2018) "Neural Ordinary Differential Equations".
    Adapted for racing scenarios where continuous dynamics and smooth trajectories
    are crucial for optimal performance.
    """

    def __init__(self, hidden_dims: List[int] = [256, 256], activation: str = "relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims
        self.activation = activation

        self.ode_layers = []
        for dim in hidden_dims:
            self.ode_layers.extend(
                [
                    layers.Dense(dim, activation=activation.value if hasattr(activation, "value") else activation),
                    layers.BatchNormalization(),
                    layers.Dropout(0.1),
                ]
            )

        self.ode_output = layers.Dense(hidden_dims[0], activation="tanh")

    def ode_func(self, t, y):
        """ODE function representing system dynamics."""
        x = y
        for layer in self.ode_layers:
            x = layer(x)
        return self.ode_output(x)

    def call(self, inputs, training=None):
        """Forward pass through Neural ODE."""
        y0 = inputs

        t = tf.linspace(0.0, 1.0, 5)

        solution = tfp.math.ode.BDF().solve(ode_fn=self.ode_func, initial_time=t[0], initial_state=y0, solution_times=t)

        return solution.states[-1]


class GraphNeuralRacingNetwork(layers.Layer):
    """Graph Neural Network for relational reasoning in racing scenarios.

    Processes relationships between track elements, other vehicles, and
    environmental features using graph convolutions. Useful for multi-agent
    racing and complex scene understanding.
    """

    def __init__(self, node_features: int = 64, edge_features: int = 32, num_layers: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_layers = num_layers

        self.node_transforms = []
        self.edge_transforms = []
        self.aggregation_layers = []

        for _ in range(num_layers):
            self.node_transforms.append(layers.Dense(node_features, activation="relu"))
            self.edge_transforms.append(layers.Dense(edge_features, activation="relu"))
            self.aggregation_layers.append(layers.Dense(node_features, activation="relu"))

    def call(self, node_features, edge_features, adjacency_matrix, training=None):
        """Forward pass through Graph Neural Network."""
        h = node_features

        for i in range(self.num_layers):
            h_transformed = self.node_transforms[i](h)

            self.edge_transforms[i](edge_features)

            messages = tf.matmul(adjacency_matrix, h_transformed)

            h_new = self.aggregation_layers[i](tf.concat([h, messages], axis=-1))

            h = h + h_new

        return h


class CapsuleRacingNetwork(layers.Layer):
    """Capsule Network for viewpoint-invariant racing perception.

    Based on Sabour et al. (2017) "Dynamic Routing Between Capsules".
    Provides robust object recognition that maintains spatial relationships
    and handles viewpoint variations common in racing scenarios.
    """

    def __init__(self, num_primary_capsules: int = 32, primary_capsule_dim: int = 8, num_routing_iterations: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.num_primary_capsules = num_primary_capsules
        self.primary_capsule_dim = primary_capsule_dim
        self.num_routing_iterations = num_routing_iterations

        self.primary_conv = layers.Conv2D(
            num_primary_capsules * primary_capsule_dim, kernel_size=9, strides=2, activation="relu", padding="valid"
        )

        self.routing_weights = self.add_weight(
            shape=(num_primary_capsules, primary_capsule_dim, 16),
            initializer="random_normal",
            trainable=True,
            name="routing_weights",
        )

    def squash(self, vectors):
        """Squashing function for capsule outputs."""
        vec_squared_norm = tf.reduce_sum(tf.square(vectors), axis=-1, keepdims=True)
        scalar_factor = vec_squared_norm / (1 + vec_squared_norm)
        return scalar_factor * vectors / tf.sqrt(vec_squared_norm + 1e-9)

    def call(self, inputs, training=None):
        """Forward pass through Capsule Network."""
        primary_capsules = self.primary_conv(inputs)

        batch_size = tf.shape(primary_capsules)[0]
        primary_capsules = tf.reshape(primary_capsules, [batch_size, -1, self.primary_capsule_dim])

        primary_capsules = self.squash(primary_capsules)

        u_hat = tf.einsum("bij,jkl->bikl", primary_capsules, self.routing_weights)

        b = tf.zeros([batch_size, self.num_primary_capsules, 1])

        for iteration in range(self.num_routing_iterations):
            c = tf.nn.softmax(b, axis=1)
            s = tf.reduce_sum(c * u_hat, axis=1)
            v = self.squash(s)

            if iteration < self.num_routing_iterations - 1:
                b += tf.reduce_sum(u_hat * tf.expand_dims(v, 1), axis=-1, keepdims=True)

        return v


class BayesianCNNRacing(layers.Layer):
    """Bayesian Convolutional Neural Network for uncertainty quantification.

    Based on Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation".
    Provides uncertainty estimates crucial for safety-critical racing decisions.
    """

    def __init__(self, filters: int = 64, dropout_rate: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.dropout_rate = dropout_rate

        self.conv1 = layers.Conv2D(filters, 3, activation="relu", padding="same")
        self.dropout1 = layers.Dropout(dropout_rate)

        self.conv2 = layers.Conv2D(filters * 2, 3, activation="relu", padding="same")
        self.dropout2 = layers.Dropout(dropout_rate)

        self.conv3 = layers.Conv2D(filters * 4, 3, activation="relu", padding="same")
        self.dropout3 = layers.Dropout(dropout_rate)

        self.global_pool = layers.GlobalAveragePooling2D()
        self.dense = layers.Dense(256, activation="relu")
        self.dropout_dense = layers.Dropout(dropout_rate)

    def call(self, inputs, training=None):
        """Forward pass with uncertainty quantification."""
        x = self.conv1(inputs)
        x = self.dropout1(x, training=True)

        x = self.conv2(x)
        x = self.dropout2(x, training=True)

        x = self.conv3(x)
        x = self.dropout3(x, training=True)

        x = self.global_pool(x)
        x = self.dense(x)
        x = self.dropout_dense(x, training=True)

        return x

    def predict_with_uncertainty(self, inputs, num_samples: int = 100):
        """Generate predictions with uncertainty estimates."""
        predictions = []
        for _ in range(num_samples):
            pred = self(inputs, training=True)
            predictions.append(pred)

        predictions = tf.stack(predictions)
        mean = tf.reduce_mean(predictions, axis=0)
        variance = tf.reduce_mean(tf.square(predictions - mean), axis=0)

        return mean, variance


class VelocityAwareNetwork(layers.Layer):
    """Network with explicit velocity and dynamics modeling.

    Incorporates vehicle dynamics and velocity information directly into
    the network architecture for improved racing performance.
    """

    def __init__(self, hidden_dims: List[int] = [256, 128], **kwargs):
        super().__init__(**kwargs)
        self.hidden_dims = hidden_dims

        self.visual_conv1 = layers.Conv2D(32, 8, strides=4, activation="relu")
        self.visual_conv2 = layers.Conv2D(64, 4, strides=2, activation="relu")
        self.visual_conv3 = layers.Conv2D(128, 3, activation="relu")
        self.visual_pool = layers.GlobalAveragePooling2D()

        self.velocity_dense1 = layers.Dense(64, activation="relu")
        self.velocity_dense2 = layers.Dense(64, activation="relu")

        self.fusion_layers = []
        for dim in hidden_dims:
            self.fusion_layers.append(layers.Dense(dim, activation="relu"))
            self.fusion_layers.append(layers.Dropout(0.1))

    def call(self, inputs, training=None):
        """Forward pass with velocity-aware processing."""
        visual_input, velocity_input = inputs

        visual_features = self.visual_conv1(visual_input)
        visual_features = self.visual_conv2(visual_features)
        visual_features = self.visual_conv3(visual_features)
        visual_features = self.visual_pool(visual_features)

        velocity_features = self.velocity_dense1(velocity_input)
        velocity_features = self.velocity_dense2(velocity_features)

        combined = tf.concat([visual_features, velocity_features], axis=-1)

        x = combined
        for layer in self.fusion_layers:
            x = layer(x, training=training)

        return x


class CrossModalAttentionFusion(layers.Layer):
    """Cross-modal attention for sensor fusion in racing.

    Implements attention mechanisms across different sensor modalities
    (camera, LiDAR, IMU) for comprehensive environmental understanding.
    """

    def __init__(self, num_heads: int = 8, key_dim: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.camera_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.lidar_attention = layers.MultiHeadAttention(num_heads, key_dim)
        self.fusion_attention = layers.MultiHeadAttention(num_heads, key_dim)

        self.camera_proj = layers.Dense(key_dim * num_heads)
        self.lidar_proj = layers.Dense(key_dim * num_heads)

        self.layer_norm1 = layers.LayerNormalization()
        self.layer_norm2 = layers.LayerNormalization()
        self.layer_norm3 = layers.LayerNormalization()

    def call(self, inputs, training=None):
        """Forward pass with cross-modal attention."""
        camera_features, lidar_features = inputs

        camera_proj = self.camera_proj(camera_features)
        lidar_proj = self.lidar_proj(lidar_features)

        camera_attended = self.camera_attention(camera_proj, camera_proj)
        camera_attended = self.layer_norm1(camera_attended + camera_proj)

        lidar_attended = self.lidar_attention(lidar_proj, lidar_proj)
        lidar_attended = self.layer_norm2(lidar_attended + lidar_proj)

        fused = self.fusion_attention(camera_attended, lidar_attended)
        fused = self.layer_norm3(fused + camera_attended)

        return fused


class EnsembleRacingNetwork(layers.Layer):
    """Ensemble of diverse networks for robust predictions.

    Combines multiple different architectures to improve robustness
    and performance in racing scenarios.
    """

    def __init__(self, num_models: int = 3, base_filters: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.num_models = num_models
        self.base_filters = base_filters

        self.models = []

        cnn_model = tf.keras.Sequential(
            [
                layers.Conv2D(base_filters, 8, strides=4, activation="relu"),
                layers.Conv2D(base_filters * 2, 4, strides=2, activation="relu"),
                layers.Conv2D(base_filters * 4, 3, activation="relu"),
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
            ]
        )
        self.models.append(cnn_model)

        resnet_model = tf.keras.Sequential(
            [
                layers.Conv2D(base_filters, 7, strides=2, padding="same"),
                ResidualBlock(base_filters),
                ResidualBlock(base_filters * 2, strides=2),
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
            ]
        )
        self.models.append(resnet_model)

        attention_model = tf.keras.Sequential(
            [
                layers.Conv2D(base_filters, 8, strides=4, activation="relu"),
                layers.Conv2D(base_filters * 2, 4, strides=2, activation="relu"),
                AttentionModule(num_heads=8, key_dim=32),
                layers.GlobalAveragePooling2D(),
                layers.Dense(256, activation="relu"),
            ]
        )
        self.models.append(attention_model)

        self.combination_layer = layers.Dense(256, activation="relu")

    def call(self, inputs, training=None):
        """Forward pass through ensemble."""
        model_outputs = []

        for model in self.models:
            output = model(inputs, training=training)
            model_outputs.append(output)

        combined = tf.stack(model_outputs, axis=1)
        combined = tf.reduce_mean(combined, axis=1)

        output = self.combination_layer(combined)

        return output, model_outputs


def create_neural_ode_racing_model(config: NetworkConfig) -> tf.keras.Model:
    """Create Neural ODE racing model."""
    inputs = layers.Input(shape=config.input_shape)

    x = layers.Conv2D(64, 8, strides=4, activation="relu")(inputs)
    x = layers.Conv2D(128, 4, strides=2, activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)

    ode_layer = NeuralODERacingNetwork(hidden_dims=[256, 256])
    x = ode_layer(x)

    if config.hidden_dims:
        for dim in config.hidden_dims:
            x = layers.Dense(dim, activation=config.activation.value)(x)
            x = layers.Dropout(config.dropout_rate)(x)

    outputs = layers.Dense(config.num_actions, activation="tanh")(x)

    model = Model(inputs, outputs, name="neural_ode_racing")
    return model


def create_bayesian_cnn_racing_model(config: NetworkConfig) -> tf.keras.Model:
    """Create Bayesian CNN racing model with uncertainty quantification."""
    inputs = layers.Input(shape=config.input_shape)

    bayesian_cnn = BayesianCNNRacing(filters=64, dropout_rate=config.dropout_rate)
    x = bayesian_cnn(inputs)

    if config.hidden_dims:
        for dim in config.hidden_dims:
            x = layers.Dense(dim, activation=config.activation.value)(x)
            x = layers.Dropout(config.dropout_rate)(x, training=True)

    mean_output = layers.Dense(config.num_actions, activation="tanh", name="action_mean")(x)
    log_var_output = layers.Dense(config.num_actions, activation="linear", name="action_log_var")(x)

    outputs = layers.Concatenate(name="bayesian_outputs")([mean_output, log_var_output])

    model = Model(inputs, outputs, name="bayesian_cnn_racing")
    return model


def create_velocity_aware_racing_model(config: NetworkConfig) -> tf.keras.Model:
    """Create velocity-aware racing model."""
    visual_input = layers.Input(shape=config.input_shape, name="visual_input")
    velocity_input = layers.Input(shape=(6,), name="velocity_input")

    velocity_aware = VelocityAwareNetwork(hidden_dims=config.hidden_dims or [256, 128])
    x = velocity_aware([visual_input, velocity_input])

    outputs = layers.Dense(config.num_actions, activation="tanh")(x)

    model = Model([visual_input, velocity_input], outputs, name="velocity_aware_racing")
    return model


def create_ensemble_racing_model(config: NetworkConfig) -> tf.keras.Model:
    """Create ensemble racing model."""
    inputs = layers.Input(shape=config.input_shape)

    ensemble = EnsembleRacingNetwork(num_models=3, base_filters=64)
    ensemble_output, individual_outputs = ensemble(inputs)

    x = ensemble_output
    if config.hidden_dims:
        for dim in config.hidden_dims:
            x = layers.Dense(dim, activation=config.activation.value)(x)
            x = layers.Dropout(config.dropout_rate)(x)

    outputs = layers.Dense(config.num_actions, activation="tanh")(x)

    model = Model(inputs, outputs, name="ensemble_racing")
    return model
