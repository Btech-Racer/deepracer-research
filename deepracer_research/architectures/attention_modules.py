from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow.keras import layers


class AttentionModule(layers.Layer):
    """Spatial Attention Module for focusing on critical racing elements.

    Based on the transformer attention mechanism from Vaswani et al. (2017),
    adapted for spatial feature maps in autonomous racing applications.
    """

    def __init__(self, num_heads: int = 8, key_dim: int = 64, dropout_rate: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate

        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=dropout_rate)
        self.layernorm = layers.LayerNormalization()

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through attention module."""
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        height, width, channels = inputs.shape[1], inputs.shape[2], inputs.shape[3]

        x = tf.reshape(inputs, [batch_size, height * width, channels])

        attn_output = self.attention(x, x, training=training)

        x = self.layernorm(x + attn_output, training=training)

        outputs = tf.reshape(x, [batch_size, height, width, channels])

        return outputs

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization."""
        config = super().get_config()
        config.update({"num_heads": self.num_heads, "key_dim": self.key_dim, "dropout_rate": self.dropout_rate})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AttentionModule":
        """Create layer from configuration for deserialization."""
        return cls(**config)


tf.keras.utils.get_custom_objects()["AttentionModule"] = AttentionModule
