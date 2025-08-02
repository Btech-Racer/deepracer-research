from typing import Any, Dict, List, Optional

import tensorflow as tf
from tensorflow.keras import layers


class MultiScaleFeatureExtractor(layers.Layer):
    """Multi-scale feature extraction layer for comprehensive perception.

    Extracts features at multiple spatial scales using parallel convolutions with
    different kernel sizes, then fuses them to capture both local details and
    global context. This approach is particularly effective for tasks requiring
    multi-scale understanding like object detection and autonomous driving.

    Parameters
    ----------
    filters : int
        Total number of output filters. Will be divided among the different scales.
    scales : List[int], optional
        List of kernel sizes for multi-scale convolutions. Default is [3, 5, 7].
    **kwargs
        Additional keyword arguments passed to the base Layer class.

    Attributes
    ----------
    filters : int
        Total number of output filters.
    scales : List[int]
        List of kernel sizes for different scales.
    scale_convs : List[tf.keras.layers.Conv2D]
        List of convolutional layers for each scale.
    fusion_conv : tf.keras.layers.Conv2D
        Fusion layer to combine multi-scale features.
    batch_norm : tf.keras.layers.BatchNormalization
        Batch normalization layer applied after fusion.

    Notes
    -----
    The multi-scale approach captures features at different receptive field sizes:
    - Small kernels (3x3) capture fine-grained local details
    - Medium kernels (5x5) capture intermediate spatial patterns
    - Large kernels (7x7) capture broader contextual information

    This design is inspired by the multi-scale processing in biological vision
    systems and has proven effective in computer vision tasks requiring
    scale-invariant feature detection.

    References
    ----------
    .. [1] Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D.,
           ... & Rabinovich, A. (2015). Going deeper with convolutions.
           In Proceedings of the IEEE conference on computer vision and pattern
           recognition (pp. 1-9).
    .. [2] Lin, T. Y., Dollár, P., Girshick, R., He, K., Hariharan, B., &
           Belongie, S. (2017). Feature pyramid networks for object detection.
           In Proceedings of the IEEE conference on computer vision and pattern
           recognition (pp. 2117-2125).
    .. [3] Zhao, H., Shi, J., Qi, X., Wang, X., & Jia, J. (2017). Pyramid scene
           parsing network. In Proceedings of the IEEE conference on computer
           vision and pattern recognition (pp. 2881-2890).
    .. [4] Chen, L. C., Papandreou, G., Kokkinos, I., Murphy, K., & Yuille, A. L.
           (2017). Deeplab: Semantic image segmentation with deep convolutional
           nets, atrous convolution, and fully connected crfs. IEEE transactions
           on pattern analysis and machine intelligence, 40(4), 834-848.
    """

    def __init__(self, filters: int, scales: Optional[List[int]] = None, **kwargs) -> None:
        """Initialize the MultiScaleFeatureExtractor layer.

        Parameters
        ----------
        filters : int
            Total number of output filters. Will be distributed among scales.
        scales : List[int], optional
            List of kernel sizes for multi-scale convolutions.
            Default is [3, 5, 7] for small, medium, and large receptive fields.
        **kwargs
            Additional keyword arguments passed to the base Layer class.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.scales = scales or [3, 5, 7]

        self.scale_convs = []
        filters_per_scale = filters // len(self.scales)

        for scale in self.scales:
            conv = layers.Conv2D(filters_per_scale, (scale, scale), padding="same", activation="relu")
            self.scale_convs.append(conv)

        self.fusion_conv = layers.Conv2D(filters, (1, 1), activation="relu")
        self.batch_norm = layers.BatchNormalization()

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """Forward pass through the multi-scale feature extractor.

        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, height, width, channels).
        training : bool, optional
            Whether the layer is in training mode. Affects batch normalization behavior.

        Returns
        -------
        tf.Tensor
            Output tensor with multi-scale features.
            Shape: (batch_size, height, width, filters).
        """
        scale_features = []
        for conv in self.scale_convs:
            scale_features.append(conv(inputs))

        concatenated = layers.Concatenate(axis=-1)(scale_features)

        fused = self.fusion_conv(concatenated)
        output = self.batch_norm(fused, training=training)

        return output

    def get_config(self) -> Dict[str, Any]:
        """Get layer configuration for serialization.

        Returns
        -------
        Dict[str, Any]
            Configuration dictionary containing layer parameters.
        """
        config = super().get_config()
        config.update({"filters": self.filters, "scales": self.scales})
        return config

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "MultiScaleFeatureExtractor":
        """Create layer from configuration for deserialization.

        Parameters
        ----------
        config : Dict[str, Any]
            Configuration dictionary containing layer parameters.

        Returns
        -------
        MultiScaleFeatureExtractor
            Reconstructed MultiScaleFeatureExtractor layer instance.
        """
        return cls(**config)


tf.keras.utils.get_custom_objects()["MultiScaleFeatureExtractor"] = MultiScaleFeatureExtractor
