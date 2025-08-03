import traceback
from typing import Any, Dict, List

import tensorflow as tf

from deepracer_research.utils.logger import error, info, warning


class ModelOptimizer:
    """Class for applying performance optimizations to neural networks."""

    @staticmethod
    def enable_mixed_precision() -> None:
        """Enable mixed precision training for faster training with less memory."""
        try:
            policy = tf.keras.mixed_precision.Policy("mixed_float16")
            tf.keras.mixed_precision.set_global_policy(policy)
            info(
                "Mixed precision training enabled",
                extra={
                    "optimization": "mixed_precision",
                    "policy": "mixed_float16",
                    "benefits": ["faster_training", "reduced_memory", "automatic_loss_scaling"],
                },
            )
        except Exception as e:
            error(
                "Failed to enable mixed precision training",
                extra={"error": str(e), "optimization": "mixed_precision", "error_details": traceback.format_exc()},
            )
            raise

    @staticmethod
    def optimize_for_inference(
        model: tf.keras.Model, quantization_type: str = "float16", optimize_for_size: bool = True
    ) -> bytes:
        """Optimize model for inference performance.

        Parameters
        ----------
        model : tf.keras.Model
            The model to optimize
        quantization_type : str, optional
            Type of quantization ('float16', 'int8', 'dynamic'), by default 'float16'
        optimize_for_size : bool, optional
            Whether to optimize for size vs speed, by default True

        Returns
        -------
        bytes
            Optimized TensorFlow Lite model
        """
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)

            if optimize_for_size:
                converter.optimizations = [tf.lite.Optimize.DEFAULT]

            if quantization_type == "float16":
                converter.target_spec.supported_types = [tf.float16]
            elif quantization_type == "int8":
                converter.target_spec.supported_types = [tf.int8]
                warning(
                    "INT8 quantization requires representative dataset",
                    extra={"quantization_type": quantization_type, "requirement": "representative_dataset"},
                )

            tflite_model = converter.convert()

            original_size_mb = model.count_params() * 4 / 1024 / 1024
            optimized_size_mb = len(tflite_model) / 1024 / 1024
            compression_ratio = original_size_mb / optimized_size_mb if optimized_size_mb > 0 else 0

            info(
                "Model optimized for inference",
                extra={
                    "original_size_mb": f"{original_size_mb:.1f}",
                    "optimized_size_mb": f"{optimized_size_mb:.1f}",
                    "compression_ratio": f"{compression_ratio:.1f}x",
                    "quantization_type": quantization_type,
                    "optimized_for": "size" if optimize_for_size else "speed",
                },
            )

            return tflite_model

        except Exception as e:
            error(
                "Failed to optimize model for inference",
                extra={
                    "error": str(e),
                    "quantization_type": quantization_type,
                    "optimize_for_size": optimize_for_size,
                    "error_details": traceback.format_exc(),
                },
            )
            raise

    @staticmethod
    def create_efficient_callbacks(
        checkpoint_path: str = "best_model.h5",
        monitor_metric: str = "val_loss",
        patience: int = 10,
        reduce_lr_patience: int = 5,
    ) -> List[tf.keras.callbacks.Callback]:
        """Create callbacks for efficient training.

        Parameters
        ----------
        checkpoint_path : str, optional
            Path to save model checkpoints, by default 'best_model.h5'
        monitor_metric : str, optional
            Metric to monitor for callbacks, by default 'val_loss'
        patience : int, optional
            Patience for early stopping, by default 10
        reduce_lr_patience : int, optional
            Patience for learning rate reduction, by default 5

        Returns
        -------
        List[tf.keras.callbacks.Callback]
            List of configured callbacks
        """
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor=monitor_metric, factor=0.5, patience=reduce_lr_patience, min_lr=1e-7, verbose=1, cooldown=2
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor=monitor_metric, patience=patience, restore_best_weights=True, verbose=1, min_delta=1e-4
            ),
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint_path, monitor=monitor_metric, save_best_only=True, save_weights_only=False, verbose=1, mode="auto"
            ),
            tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * tf.math.exp(-0.1 * epoch), verbose=0),
            tf.keras.callbacks.TensorBoard(
                log_dir="./logs", histogram_freq=1, write_graph=True, write_images=False, update_freq="epoch"
            ),
        ]

        info(
            "Efficient training callbacks configured",
            extra={
                "num_callbacks": len(callbacks),
                "checkpoint_path": checkpoint_path,
                "monitor_metric": monitor_metric,
                "patience": patience,
                "reduce_lr_patience": reduce_lr_patience,
            },
        )

        return callbacks

    @staticmethod
    def setup_efficient_data_pipeline(
        batch_size: int = 32,
        prefetch_buffer: int = tf.data.AUTOTUNE,
        shuffle_buffer: int = 1000,
        cache_data: bool = True,
        num_parallel_calls: int = tf.data.AUTOTUNE,
    ) -> Dict[str, Any]:
        """Setup efficient data pipeline configuration.

        Parameters
        ----------
        batch_size : int, optional
            Batch size for training, by default 32
        prefetch_buffer : int, optional
            Prefetch buffer size, by default tf.data.AUTOTUNE
        shuffle_buffer : int, optional
            Shuffle buffer size, by default 1000
        cache_data : bool, optional
            Whether to cache data in memory, by default True
        num_parallel_calls : int, optional
            Number of parallel calls for data processing, by default tf.data.AUTOTUNE

        Returns
        -------
        Dict[str, Any]
            Data pipeline configuration
        """
        config = {
            "batch_size": batch_size,
            "prefetch_buffer": prefetch_buffer,
            "num_parallel_calls": num_parallel_calls,
            "cache": cache_data,
            "shuffle_buffer": shuffle_buffer,
            "drop_remainder": True,
            "deterministic": False,
        }

        info(
            "Efficient data pipeline configured",
            extra={
                "batch_size": batch_size,
                "prefetch_buffer": str(prefetch_buffer),
                "shuffle_buffer": shuffle_buffer,
                "cache_enabled": cache_data,
                "parallel_calls": str(num_parallel_calls),
            },
        )

        return config

    @staticmethod
    def optimize_model_architecture(model: tf.keras.Model) -> tf.keras.Model:
        """Apply architectural optimizations to the model.

        Parameters
        ----------
        model : tf.keras.Model
            The model to optimize

        Returns
        -------
        tf.keras.Model
            Optimized model
        """
        try:
            model.compile(optimizer=model.optimizer, loss=model.loss, metrics=model.metrics, jit_compile=True)

            info(
                "Model architecture optimized",
                extra={
                    "optimization": "xla_compilation",
                    "total_params": model.count_params(),
                    "trainable_params": sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
                },
            )

            return model

        except Exception as e:
            warning("Could not enable XLA compilation", extra={"error": str(e), "fallback": "using_standard_compilation"})
            return model

    @staticmethod
    def get_optimization_recommendations(model: tf.keras.Model) -> Dict[str, Any]:
        """Get optimization recommendations based on model characteristics.

        Parameters
        ----------
        model : tf.keras.Model
            The model to analyze

        Returns
        -------
        Dict[str, Any]
            Optimization recommendations
        """
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

        recommendations = {
            "mixed_precision": total_params > 1_000_000,
            "model_pruning": trainable_params > 500_000,
            "quantization": True,
            "xla_compilation": True,
            "gradient_checkpointing": total_params > 10_000_000,
        }

        size_category = "small"
        if total_params > 10_000_000:
            size_category = "very_large"
        elif total_params > 1_000_000:
            size_category = "large"
        elif total_params > 100_000:
            size_category = "medium"

        recommendations["size_category"] = size_category

        info(
            "Model optimization analysis completed",
            extra={
                "total_params": total_params,
                "trainable_params": trainable_params,
                "size_category": size_category,
                "recommendations": recommendations,
            },
        )

        return recommendations
