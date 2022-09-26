# Tensorflow
import tensorflow as tf

# Local
from data_augmentations.tfda_2d.base import TFDABase
from data_augmentations.tfda_2d.defs import TFbT, TFDAData


class DataChannelSelectionTransform(TFDABase):
    """Selects color channels from the batch and discards the others."""

    def __init__(self, channels: tf.Tensor, **kws: tf.Tensor) -> None:
        super().__init__(**kws)
        self.channels = channels

    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""

        return dataset.new_data(tf.gather(dataset.data, self.channels, axis=1))


class SegChannelSelectionTransform(TFDABase):
    """Segmentations may have more than one channel.

    This transform selects segmentation channels.
    """

    def __init__(
        self,
        channels: tf.Tensor,
        keep_discarded: tf.Tensor = TFbT,
        **kws: tf.Tensor
    ) -> None:
        super().__init__(**kws)
        self.channels = channels
        self.keep_discarded = keep_discarded

    @tf.function(experimental_follow_type_hints=True)
    def call(self, dataset: TFDAData) -> TFDAData:
        """Call the transform."""
        seg = dataset.seg

        if tf.math.reduce_any(tf.math.is_nan(seg)):
            tf.get_logger().warn(
                "You used SegChannelSelectionTransform but "
                "there is no 'seg' key in your data_dict, returning "
                "data_dict unmodified",
            )
            seg = seg
        else:
            # TODO: keep_discarded

            seg = tf.gather(seg, self.channels, axis=1)
        return TFDAData(dataset.data, seg)


if __name__ == "__main__":
    dataset = next(
        iter(
            tf.data.Dataset.range(
                8 * 1 * 1 * 40 * 56 * 40, output_type=tf.float32
            )
            .batch(40)
            .batch(56)
            .batch(40)
            .batch(2)
            .batch(4)
            .prefetch(4)
        )
    )

    with tf.device("/CPU:0"):

        # dcst = DataChannelSelectionTransform(to_tf_float([]))
        # tf.print(dcst(dict(data=dataset))["data"].shape)

        scst = SegChannelSelectionTransform(tf.cast([0, 1], tf.int64))
        tf.print(scst(TFDAData(seg=dataset))["seg"].shape)
        tf.print(
            SegChannelSelectionTransform(tf.cast([0], tf.int64))(
                TFDAData(seg=dataset)
            )["seg"].shape
        )
