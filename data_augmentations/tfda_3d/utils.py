# Tensorflow
import tensorflow as tf

# Types
from typing import Any


@tf.function
def to_tf_bool(x: Any) -> tf.Tensor:
    """Convert python bool to tf bool."""
    return tf.cast(x, tf.bool)


@tf.function
def to_tf_float(x: Any) -> tf.Tensor:
    """Convert python bool to tf float32."""
    return tf.cast(x, tf.float32)


@tf.function
def to_tf_int(x: Any) -> tf.Tensor:
    """Convert python bool to tf int64."""
    return tf.cast(x, tf.int64)


@tf.function(experimental_follow_type_hints=True)
def isnan(x: tf.Tensor) -> tf.Tensor:
    """Check if there is any nan in it."""
    return tf.math.reduce_any(tf.math.is_nan(x))


@tf.function(experimental_follow_type_hints=True)
def isnotnan(x: tf.Tensor) -> tf.Tensor:
    """Check if there is any nan in it."""
    return tf.math.logical_not(tf.math.reduce_any(tf.math.is_nan(x)))
