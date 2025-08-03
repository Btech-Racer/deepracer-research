from typing import Any, Dict

import tensorflow as tf

from deepracer_research.utils.logger import error, info, warning


def validate_tensorflow_gpu(test_computation: bool = True) -> bool:
    """Validate TensorFlow GPU setup.

    Parameters
    ----------
    test_computation : bool, optional
        Whether to test actual GPU computation, by default True

    Returns
    -------
    bool
        True if GPU is available and properly configured
    """
    try:
        gpus = tf.config.experimental.list_physical_devices("GPU")

        if not gpus:
            info("No GPUs detected, using CPU", extra={"gpu_available": False, "device_count": 0})
            return False

        if test_computation:
            with tf.device("/GPU:0"):
                a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
                b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
                c = tf.matmul(a, b)

                _ = c.numpy()

        info(
            "TensorFlow GPU validation successful",
            extra={"gpu_available": True, "device_count": len(gpus), "test_computation": test_computation},
        )

        return True

    except Exception as e:
        error(
            "TensorFlow GPU validation failed",
            extra={"error": str(e), "gpu_available": False, "test_computation": test_computation},
        )
        return False


class GPUOptimizer:
    """GPU-specific optimization utilities for neural network training."""

    @staticmethod
    def configure_gpu_memory_growth() -> bool:
        """Configure GPU memory growth to avoid out-of-memory errors.

        Returns
        -------
        bool
            True if configuration was successful
        """
        gpus = tf.config.experimental.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)

                info(
                    "GPU memory growth enabled",
                    extra={
                        "num_gpus": len(gpus),
                        "memory_growth": True,
                        "benefits": ["dynamic_allocation", "multi_process_support"],
                    },
                )
                return True

            except RuntimeError as e:
                warning("Could not set GPU memory growth", extra={"error": str(e), "num_gpus": len(gpus)})
                return False
        else:
            info("No GPUs detected, using CPU", extra={"gpu_available": False})
            return False

    @staticmethod
    def set_memory_limit(memory_limit_mb: int, gpu_index: int = 0) -> bool:
        """Set a memory limit for a specific GPU.

        Parameters
        ----------
        memory_limit_mb : int
            Memory limit in megabytes
        gpu_index : int, optional
            GPU index to configure, by default 0

        Returns
        -------
        bool
            True if configuration was successful
        """
        gpus = tf.config.experimental.list_physical_devices("GPU")

        if not gpus or gpu_index >= len(gpus):
            error("Invalid GPU index or no GPUs available", extra={"gpu_index": gpu_index, "available_gpus": len(gpus)})
            return False

        try:
            tf.config.experimental.set_memory_growth(gpus[gpu_index], False)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[gpu_index], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory_limit_mb)]
            )

            info("GPU memory limit configured", extra={"gpu_index": gpu_index, "memory_limit_mb": memory_limit_mb})
            return True

        except RuntimeError as e:
            error(
                "Failed to set GPU memory limit",
                extra={"error": str(e), "gpu_index": gpu_index, "memory_limit_mb": memory_limit_mb},
            )
            return False

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        """Get comprehensive information about available GPUs.

        Returns
        -------
        Dict[str, Any]
            Detailed GPU information
        """
        gpus = tf.config.experimental.list_physical_devices("GPU")

        gpu_info = {
            "num_gpus": len(gpus),
            "gpu_names": [gpu.name for gpu in gpus],
            "cuda_available": len(gpus) > 0,
            "tensorflow_version": tf.__version__,
        }

        if gpu_info["cuda_available"]:
            try:
                cuda_version = tf.test.is_built_with_cuda()
                gpu_info["cuda_built"] = cuda_version

                gpu_details = []
                for i, gpu in enumerate(gpus):
                    try:
                        memory_info = tf.config.experimental.get_memory_info(f"GPU:{i}")
                        gpu_details.append(
                            {
                                "index": i,
                                "name": gpu.name,
                                "current_memory_mb": memory_info["current"] / 1024 / 1024,
                                "peak_memory_mb": memory_info["peak"] / 1024 / 1024,
                            }
                        )
                    except:
                        gpu_details.append({"index": i, "name": gpu.name, "memory_info": "unavailable"})

                gpu_info["gpu_details"] = gpu_details

            except Exception as e:
                gpu_info["cuda_info_error"] = str(e)

        if gpu_info["cuda_available"]:
            info(
                "GPU information gathered",
                extra={
                    "num_gpus": gpu_info["num_gpus"],
                    "tensorflow_version": gpu_info["tensorflow_version"],
                    "cuda_built": gpu_info.get("cuda_built", "unknown"),
                },
            )

            for i, name in enumerate(gpu_info["gpu_names"]):
                info(f"   GPU {i}: {name}")
        else:
            info("Using CPU for training", extra={"gpu_available": False, "tensorflow_version": gpu_info["tensorflow_version"]})

        return gpu_info

    @staticmethod
    def benchmark_gpu_performance() -> Dict[str, float]:
        """Benchmark GPU performance with simple operations.

        Returns
        -------
        Dict[str, float]
            Performance metrics
        """
        if not validate_tensorflow_gpu(test_computation=False):
            warning("No GPU available for benchmarking")
            return {}

        try:
            with tf.device("/GPU:0"):
                size = 1000
                a = tf.random.normal([size, size])
                b = tf.random.normal([size, size])

                for _ in range(3):
                    _ = tf.matmul(a, b)

                import time

                start_time = time.time()
                for _ in range(10):
                    c = tf.matmul(a, b)
                    _ = c.numpy()
                end_time = time.time()

                matmul_time = (end_time - start_time) / 10

                x = tf.random.normal([32, 224, 224, 3])
                conv_layer = tf.keras.layers.Conv2D(64, 3, activation="relu")

                for _ in range(3):
                    _ = conv_layer(x)

                start_time = time.time()
                for _ in range(10):
                    y = conv_layer(x)
                    _ = y.numpy()
                end_time = time.time()

                conv_time = (end_time - start_time) / 10

            benchmark_results = {
                "matrix_multiplication_ms": matmul_time * 1000,
                "convolution_ms": conv_time * 1000,
                "matrix_size": size,
                "conv_input_shape": [32, 224, 224, 3],
            }

            info("GPU benchmark completed", extra=benchmark_results)

            return benchmark_results

        except Exception as e:
            error("GPU benchmark failed", extra={"error": str(e)})
            return {}

    @staticmethod
    def validate_setup() -> bool:
        """Validate complete TensorFlow GPU setup.

        Returns
        -------
        bool
            True if GPU is available and properly configured
        """
        return validate_tensorflow_gpu(test_computation=True)

    @staticmethod
    def optimize_for_training() -> bool:
        """Apply optimal GPU settings for training.

        Returns
        -------
        bool
            True if optimization was successful
        """
        success = True

        if not GPUOptimizer.configure_gpu_memory_growth():
            success = False

        try:
            tf.debugging.set_log_device_placement(False)

            tf.config.optimizer.set_jit(True)

            info(
                "GPU training optimizations applied",
                extra={"memory_growth": True, "xla_enabled": True, "device_placement_logging": False},
            )

        except Exception as e:
            warning("Some GPU optimizations failed", extra={"error": str(e)})
            success = False

        return success
