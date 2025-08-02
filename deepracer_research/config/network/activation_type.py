from enum import Enum


class ActivationType(str, Enum):
    """Supported activation function types for neural networks

    References
    ----------
        GLOROT, Xavier, Antoine BORDES and Yoshua BENGIO. Deep sparse rectifier
        neural networks. In: Proceedings of the fourteenth international conference
        on artificial intelligence and statistics. 2011, pp. 315-323.

        HENDRYCKS, Dan and Kevin GIMPEL. Gaussian error linear units (gelus)
        [online]. 2016. arXiv preprint arXiv:1606.08415. [visited on July 30, 2025].
        Available from: https://arxiv.org/abs/1606.08415

        RAMACHANDRAN, Prajit, Barret ZOPH and Quoc V. LE. Searching for activation
        functions [online]. 2017. arXiv preprint arXiv:1710.05941. [visited on July 30, 2025].
        Available from: https://arxiv.org/abs/1710.05941
    """

    RELU = "relu"
    """Rectified Linear Unit (ReLU): f(x) = max(0, x)

    Most commonly used activation function. Simple and computationally efficient.
    Suffers from "dying ReLU" problem where neurons can become inactive.
    Good for: Hidden layers in deep networks, CNNs.

    References
    ----------
        NAIR, Vinod and Geoffrey E. HINTON. Rectified linear units improve restricted
        boltzmann machines. In: Proceedings of the 27th international conference on
        machine learning (ICML-10). 2010, pp. 807-814.
    """

    LEAKY_RELU = "leaky_relu"
    """Leaky ReLU: f(x) = max(αx, x) where α is small (typically 0.01)

    Addresses the dying ReLU problem by allowing small negative values.
    Helps maintain gradient flow for negative inputs.
    Good for: Deep networks where dying ReLU is a concern.

    References
    ----------
        MAAS, Andrew L., Awni Y. HANNUN and Andrew Y. NG. Rectifier nonlinearities
        improve neural network acoustic models. In: Proc. icml. 2013, vol. 30, no. 1, p. 3.
    """

    ELU = "elu"
    """Exponential Linear Unit: f(x) = x if x > 0, else α(e^x - 1)

    Smooth function that can produce negative outputs. Has faster convergence
    than ReLU and reduces bias shift. More computationally expensive than ReLU.
    Good for: Networks requiring smooth gradients and faster convergence.

    References
    ----------
        CLEVERT, Djork-Arné, Thomas UNTERTHINER and Sepp HOCHREITER. Fast and accurate
        deep network learning by exponential linear units (elus) [online]. 2015.
        arXiv preprint arXiv:1511.07289. [visited on July 30, 2025].
        Available from: https://arxiv.org/abs/1511.07289
    """

    SELU = "selu"
    """Scaled ELU: Self-normalizing activation function

    Special case of ELU with specific α and scale values that enable
    self-normalization properties. Maintains mean ≈ 0 and variance ≈ 1.
    Good for: Deep feedforward networks, self-normalizing networks.

    References
    ----------
        KLAMBAUER, Günter, Thomas UNTERTHINER, Andreas MAYR and Sepp HOCHREITER.
        Self-normalizing neural networks. In: Advances in neural information
        processing systems. 2017, pp. 971-980.
    """

    SWISH = "swish"
    """Swish: f(x) = x * sigmoid(x)

    Smooth, non-monotonic activation function developed by Google.
    Often outperforms ReLU on deeper models. Computationally more expensive.
    Good for: Deep networks, image classification, NLP tasks.

    References
    ----------
        RAMACHANDRAN, Prajit, Barret ZOPH and Quoc V. LE. Searching for activation
        functions [online]. 2017. arXiv preprint arXiv:1710.05941. [visited on July 30, 2025].
        Available from: https://arxiv.org/abs/1710.05941
    """

    MISH = "mish"
    """Mish: f(x) = x * tanh(softplus(x))

    Smooth, non-monotonic activation with unbounded above, bounded below.
    Often provides better accuracy than Swish and ReLU on various tasks.
    Good for: Image classification, object detection, segmentation.

    References
    ----------
        MISRA, Diganta. Mish: A self regularized non-monotonic neural activation
        function [online]. 2019. arXiv preprint arXiv:1908.08681. [visited on July 30, 2025].
        Available from: https://arxiv.org/abs/1908.08681
    """

    GELU = "gelu"
    """Gaussian Error Linear Unit: f(x) = x * Φ(x)

    Smooth approximation combining properties of ReLU and dropout.
    Widely used in transformer architectures (BERT, GPT).
    Good for: Transformer models, NLP tasks, modern architectures.

    References
    ----------
        HENDRYCKS, Dan and Kevin GIMPEL. Gaussian error linear units (gelus)
        [online]. 2016. arXiv preprint arXiv:1606.08415. [visited on July 30, 2025].
        Available from: https://arxiv.org/abs/1606.08415
    """

    TANH = "tanh"
    """Hyperbolic Tangent: f(x) = (e^x - e^-x) / (e^x + e^-x)

    S-shaped curve, outputs between -1 and 1. Zero-centered output.
    Can suffer from vanishing gradient problem in deep networks.
    Good for: Shallow networks, RNNs, when zero-centered output is needed.

    References
    ----------
        LECUN, Yann, Léon BOTTOU, Genevieve B. ORR and Klaus-Robert MÜLLER.
        Efficient backprop. In: Neural networks: Tricks of the trade. Springer, 1998, pp. 9-50.
    """

    SIGMOID = "sigmoid"
    """Sigmoid: f(x) = 1 / (1 + e^-x)

    S-shaped curve, outputs between 0 and 1. Historically popular but
    suffers from vanishing gradients and is not zero-centered.
    Good for: Binary classification output layer, gates in RNNs/LSTMs.

    References
    ----------
        RUMELHART, David E., Geoffrey E. HINTON and Ronald J. WILLIAMS. Learning
        representations by back-propagating errors. Nature. 1986, vol. 323, no. 6088,
        pp. 533-536.
    """

    SOFTMAX = "softmax"
    """Softmax: f(x_i) = e^x_i / Σ(e^x_j)

    Converts vector of real numbers into probability distribution.
    Output values sum to 1. Used for multi-class classification.
    Good for: Multi-class classification output layer, attention mechanisms.

    References
    ----------
        BRIDLE, John S. Probabilistic interpretation of feedforward classification
        network outputs, with relationships to statistical pattern recognition.
        In: Neurocomputing. Springer, 1990, pp. 227-236.
    """

    LINEAR = "linear"
    """Linear/Identity: f(x) = x

    No transformation applied. Passes input directly as output.
    Used when no non-linearity is desired.
    Good for: Regression output layers, skip connections.

    References
    ----------
        HORNIK, Kurt. Approximation capabilities of multilayer feedforward networks.
        Neural networks. 1991, vol. 4, no. 2, pp. 251-257.
    """

    RELU6 = "relu6"
    """ReLU6: f(x) = min(max(0, x), 6)

    ReLU capped at 6. Helps with numerical stability and is hardware-friendly
    for mobile/embedded deployments. Used in MobileNet architectures.
    Good for: Mobile/embedded models, low-precision training.

    References
    ----------
        HOWARD, Andrew G., Menglong ZHU, Bo CHEN, Dmitry KALENICHENKO, Weijun WANG,
        Tobias WEYAND, Marco ANDREETTO and Hartwig ADAM. Mobilenets: Efficient
        convolutional neural networks for mobile vision applications [online]. 2017.
        arXiv preprint arXiv:1704.04861. [visited on July 30, 2025].
        Available from: https://arxiv.org/abs/1704.04861
    """

    HARD_SIGMOID = "hard_sigmoid"
    """Hard Sigmoid: Piecewise linear approximation of sigmoid

    Computationally efficient approximation: f(x) = max(0, min(1, (x + 1)/2))
    Faster than standard sigmoid while maintaining similar properties.
    Good for: Mobile/embedded applications, when computational efficiency is critical.

    References
    ----------
        COURBARIAUX, Matthieu, Yoshua BENGIO and Jean-Pierre DAVID. Binaryconnect:
        Training deep neural networks with binary weights during propagations.
        In: Advances in neural information processing systems. 2015, pp. 3123-3131.
    """

    EXPONENTIAL = "exponential"
    """Exponential: f(x) = e^x

    Unbounded exponential growth. Can lead to numerical instability
    with large inputs. Use with caution and proper input scaling.
    Good for: Specialized applications, Poisson regression output layers.

    References
    ----------
        MCCULLAGH, Peter and John A. NELDER. Generalized linear models. 2nd ed.
        CRC press, 1989. vol. 37.
    """
