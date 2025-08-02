from typing import Any, Dict, Optional

import tensorflow as tf
from tensorflow.keras import layers


class ResidualBlock(layers.Layer):
    """Residual Block with skip connections for deep feature learning

    Parameters
    ----------
    filters : int
        Number of filters (channels) in the convolutional layers.
    strides : int, default=1
        Stride size for the first convolutional layer. When > 1, performs downsampling.
    dropout_rate : float, default=0.1
        Dropout rate applied between convolutional layers for regularization.
    **kwargs
        Additional keyword arguments passed to the base Layer class.

    Attributes
    ----------
    filters : int
        Number of filters in the convolutional layers.
    strides : int
        Stride size for the first convolution.
    dropout_rate : float
        Dropout rate for regularization.
    conv1 : tf.keras.layers.Conv2D
        First convolutional layer.
    bn1 : tf.keras.layers.BatchNormalization
        Batch normalization after first convolution.
    relu1 : tf.keras.layers.ReLU
        ReLU activation after first batch normalization.
    conv2 : tf.keras.layers.Conv2D
        Second convolutional layer.
    bn2 : tf.keras.layers.BatchNormalization
        Batch normalization after second convolution.
    dropout : tf.keras.layers.Dropout
        Dropout layer for regularization.
    shortcut : tf.keras.layers.Conv2D, optional
        Shortcut connection conv layer when input/output dimensions differ.
    shortcut_bn : tf.keras.layers.BatchNormalization, optional
        Batch normalization for shortcut connection.
    relu_final : tf.keras.layers.ReLU
        Final ReLU activation after residual addition.

    References
    ----------
    .. [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning
           for image recognition. In Proceedings of the IEEE conference on
           computer vision and pattern recognition (pp. 770-778).
    .. [2] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in
           deep residual networks. In European conference on computer vision
           (pp. 630-645). Springer.
    .. [3] Szegedy, C., Ioffe, S., Vanhoucke, V., & Alemi, A. A. (2017).
           Inception-v4, inception-resnet and the impact of residual connections
           on learning. In Thirty-first AAAI conference on artificial intelligence.
    """

    def __init__(self, filters: int, strides: int = 1, dropout_rate: float = 0.1, **kwargs) -> None:
        """Initialize the ResidualBlock layer.

        Parameters
        ----------
        filters : int
            Number of filters (channels) in the convolutional layers.
        strides : int, default=1
            Stride size for the first convolutional layer.
        dropout_rate : float, default=0.1
            Dropout rate for regularization.
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.strides = strides
        self.dropout_rate = dropout_rate

        self.conv1 = layers.Conv2D(filters, (3, 3), strides=strides, padding="same")
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()

        self.conv2 = layers.Conv2D(filters, (3, 3), strides=1, padding="same")
        self.bn2 = layers.BatchNormalization()

        self.dropout = layers.Dropout(dropout_rate)

        self.shortcut = None
        self.shortcut_bn = None
        if strides != 1:
            self.shortcut = layers.Conv2D(filters, (1, 1), strides=strides, padding="same")
            self.shortcut_bn = layers.BatchNormalization()

        self.relu_final = layers.ReLU()

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through the residual block.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, height, width, channels).
        training : bool, optional
            Whether the layer is in training mode. Affects batch normalization
            and dropout behavior.

        Returns
        -------
        tf.Tensor
            Output tensor after applying residual block transformation.
            Shape: (batch_size, height//strides, width//strides, filters).
        """
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu1(x)
        x = self.dropout(x, training=training)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        shortcut = inputs
        if self.shortcut is not None:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)

        x = x + shortcut
        x = self.relu_final(x)

        return x

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary containing layer parameters.
        """
        config = super().get_config()
        config.update({"filters": self.filters, "strides": self.strides, "dropout_rate": self.dropout_rate})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ResidualBlock":
        """Create layer from configuration for deserialization.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing layer parameters.

        Returns
        -------
        ResidualBlock
            Reconstructed ResidualBlock layer instance.
        """
        return cls(**config)


tf.keras.utils.get_custom_objects()["ResidualBlock"] = ResidualBlock
