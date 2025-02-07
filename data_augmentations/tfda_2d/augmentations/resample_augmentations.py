# Tensorflow
import tensorflow as tf

# Local
from data_augmentations.tfda_2d.defs import TFbT, nan
from data_augmentations.tfda_2d.utils import isnan, isnotnan


@tf.function
def augment_liner_help(
    target_shape: tf.Tensor,
    dim: tf.Tensor,
    shp: tf.Tensor,
    ignore_axes: tf.Tensor,
):
    return tf.map_fn(
        lambda d: tf.cond(
            tf.math.reduce_any(tf.equal(ignore_axes, d)),
            lambda: shp[tf.cast(d, tf.int64)],
            lambda: target_shape[tf.cast(d, tf.int64)],
        ),
        tf.range(dim, dtype=tf.float32),
    )


@tf.function(experimental_follow_type_hints=True)
def augment_linear_downsampling_scipy(
    data_sample: tf.Tensor,
    zoom_range: tf.Tensor = (0.5, 1),
    per_channel: tf.Tensor = TFbT,
    p_per_channel: tf.Tensor = 1.0,
    channels: tf.Tensor = nan,
    order_downsample: tf.Tensor = 1,
    order_upsample: tf.Tensor = 0,
    ignore_axes: tf.Tensor = nan,
):
    """
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel: probability for downsampling/upsampling a channel

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:

        ignore_axes: tuple/list

    """
    shp = tf.shape(data_sample, out_type=tf.int64)[1:]

    target_shape = tf.cast(
        tf.round(
            tf.cast(shp, tf.float32)
            * tf.random.uniform((), zoom_range[0], zoom_range[1])
        ),
        tf.int64,
    )

    channels = tf.cast(
        tf.cond(
            isnan(channels),
            lambda: tf.range(
                tf.shape(data_sample)[0], dtype=tf.float32
            ),  # [0, 1]
            lambda: channels,
        ),
        tf.int64,
    )

    return tf.map_fn(
        lambda c: tf.cond(
            tf.less(tf.random.uniform(()), p_per_channel),
            lambda: volume_resize(
                volume_resize(
                    data_sample[c],
                    tf.cond(
                        per_channel,
                        lambda: tf.cond(
                            isnotnan(tf.cast(ignore_axes, tf.float32)),
                            lambda: tf.map_fn(
                                lambda i: tf.cond(
                                    tf.math.reduce_any(
                                        tf.equal(
                                            i, tf.cast(ignore_axes, tf.int64)
                                        )
                                    ),
                                    lambda: shp[i],
                                    lambda: target_shape[i],
                                ),
                                tf.range(
                                    tf.shape(target_shape)[0],
                                    dtype=target_shape.dtype,
                                ),
                            ),
                            lambda: tf.cast(
                                tf.round(
                                    tf.cast(shp, tf.float32)
                                    * tf.random.uniform(
                                        (), zoom_range[0], zoom_range[1]
                                    )
                                ),
                                tf.int64,
                            ),
                        ),
                        lambda: target_shape,
                    ),
                    method="nearest",
                ),
                shp,
                # NOTE: tpu doesn't support bicubic
                # method="bilinear",
                method="bicubic",
            ),
            lambda: data_sample[c],
        ),
        channels,
        fn_output_signature=tf.float32,
    )


@tf.function(experimental_follow_type_hints=True)
def volume_resize(input_data: tf.Tensor, target_shape: tf.Tensor, method: str):
    target_shape = tf.cast(target_shape, tf.int32)
    image = tf.transpose(input_data, perm=[1, 2, 0])
    image = tf.image.resize(image, target_shape[1:], method=method)
    image = tf.transpose(image, perm=[2, 0, 1])
    image = tf.image.resize(image, target_shape[:-1], method=method)
    return image


@tf.function(experimental_follow_type_hints=True)
def augment_linear_downsampling_scipy_2D(
    data_sample: tf.Tensor,
    zoom_range: tf.Tensor = (0.5, 1),
    per_channel: tf.Tensor = TFbT,
    p_per_channel: tf.Tensor = 1.0,
    channels: tf.Tensor = nan,
    order_downsample: tf.Tensor = 1,
    order_upsample: tf.Tensor = 0,
    ignore_axes: tf.Tensor = nan,
):
    """
    Downsamples each sample (linearly) by a random factor and upsamples to original resolution again (nearest neighbor)

    Info:
    * Uses scipy zoom for resampling. A bit faster than nilearn.
    * Resamples all dimensions (channels, x, y, z) with same downsampling factor (like isotropic=True from
    linear_downsampling_generator_nilearn)

    Args:
        zoom_range: can be either tuple/list/np.ndarray or tuple of tuple. If tuple/list/np.ndarray, then the zoom
        factor will be sampled from zoom_range[0], zoom_range[1] (zoom < 0 = downsampling!). If tuple of tuple then
        each inner tuple will give a sampling interval for each axis (allows for different range of zoom values for
        each axis

        p_per_channel: probability for downsampling/upsampling a channel

        per_channel (bool): whether to draw a new zoom_factor for each channel or keep one for all channels

        channels (list, tuple): if None then all channels can be augmented. If list then only the channel indices can
        be augmented (but may not always be depending on p_per_channel)

        order_downsample:

        order_upsample:

        ignore_axes: tuple/list

    """
    shp = tf.shape(data_sample, out_type=tf.int64)[1:]

    target_shape = tf.cast(
        tf.round(
            tf.cast(shp, tf.float32)
            * tf.random.uniform((), zoom_range[0], zoom_range[1])
        ),
        tf.int64,
    )

    channels = tf.cast(
        tf.cond(
            isnan(channels),
            lambda: tf.range(
                tf.shape(data_sample)[0], dtype=tf.float32
            ),  # [0, 1]
            lambda: channels,
        ),
        tf.int64,
    )

    return tf.map_fn(
        lambda c: tf.cond(
            tf.less(tf.random.uniform(()), p_per_channel),
            lambda: volume_resize_2D(
                volume_resize_2D(
                    data_sample[c],
                    tf.cond(
                        per_channel,
                        lambda: tf.cond(
                            isnotnan(tf.cast(ignore_axes, tf.float32)),
                            lambda: tf.map_fn(
                                lambda i: tf.cond(
                                    tf.math.reduce_any(
                                        tf.equal(
                                            i, tf.cast(ignore_axes, tf.int64)
                                        )
                                    ),
                                    lambda: shp[i],
                                    lambda: target_shape[i],
                                ),
                                tf.range(
                                    tf.shape(target_shape)[0],
                                    dtype=target_shape.dtype,
                                ),
                            ),
                            lambda: tf.cast(
                                tf.round(
                                    tf.cast(shp, tf.float32)
                                    * tf.random.uniform(
                                        (), zoom_range[0], zoom_range[1]
                                    )
                                ),
                                tf.int64,
                            ),
                        ),
                        lambda: target_shape,
                    ),
                    method="nearest",
                ),
                shp,
                # NOTE: tpu doesn't support bicubic
                # method="bilinear",
                method="bicubic",
            ),
            lambda: data_sample[c],
        ),
        channels,
        fn_output_signature=tf.float32,
    )



@tf.function(experimental_follow_type_hints=True)
def volume_resize_2D(input_data: tf.Tensor, target_shape: tf.Tensor, method: str):
    target_shape = tf.cast(target_shape, tf.int32)
    input_data = input_data[tf.newaxis]
    image = tf.transpose(input_data, perm=[1, 2, 0])
    image = tf.image.resize(image, target_shape, method=method)
    image = tf.transpose(image, perm=[2, 0, 1])
    return image[0]



if __name__ == '__main__':
    with tf.device("/CPU:0"):
        data = tf.random.uniform((1, 376, 376))

        res = volume_resize_2D(data[0], [350, 350], "bicubic")
        tf.print(res.shape)
