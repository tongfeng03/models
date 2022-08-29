# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser and processing for 3D segmentation datasets."""

from typing import Any, Dict, Sequence, Tuple
import tensorflow as tf
from official.vision.beta.dataloaders import decoder
from official.vision.beta.dataloaders import parser
import numpy as np

from official.projects.volumetric_models.data_augmentations.tfda_3d.transforms.spatial_transforms import SpatialTransform, MirrorTransform, SpatialTransform2D
from official.projects.volumetric_models.data_augmentations.tfda_3d.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform
from official.projects.volumetric_models.data_augmentations.tfda_3d.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from official.projects.volumetric_models.data_augmentations.tfda_3d.transforms.custom_transforms import MaskTransform, OneHotTransform, Convert3DTo2DTransform, Convert2DTo3DTransform
from official.projects.volumetric_models.data_augmentations.tfda_3d.transforms.utility_transforms import RemoveLabelTransform
from official.projects.volumetric_models.data_augmentations.tfda_3d.transforms.resample_transforms import SimulateLowResolutionTransform
from official.projects.volumetric_models.data_augmentations.tfda_3d.defs import TFDAData, TFDADefault3DParams, DTFT, TFbF, TFbT, nan, pi
from official.projects.volumetric_models.data_augmentations.tfda_3d.data_processing_utils import get_batch_size, update_tf_channel


class Decoder(decoder.Decoder):
  """A tf.Example decoder for segmentation task."""

  def __init__(self,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label',
               image_shape_key: str = 'image_shape',
               label_shape_key: str = 'label_shape'
               ):
    self._keys_to_features = {
        image_field_key: tf.io.VarLenFeature(dtype=tf.float32),
        label_field_key: tf.io.VarLenFeature(dtype=tf.int64),
        image_shape_key: tf.io.FixedLenFeature([4], tf.int64),
        label_shape_key: tf.io.FixedLenFeature([3], tf.int64)
    }

  def decode(self, serialized_example: tf.string) -> Dict[str, tf.Tensor]:
    return tf.io.parse_single_example(serialized_example,
                                      self._keys_to_features)


class Parser(parser.Parser):
  """Parser to parse an image and its annotations into a dictionary of tensors."""

  def __init__(self,
               input_size: Sequence[int] = [40, 56, 40],
               num_classes: int = 3,
               num_channels: int = 1,
               image_field_key: str = 'image/encoded',
               label_field_key: str = 'image/class/label',
               image_shape_key: str = 'image_shape',
               label_shape_key: str = 'label_shape',
               dtype: str = 'float32',
               label_dtype: str = 'int32'):
    """Initializes parameters for parsing annotations in the dataset.

    Args:
      input_size: The input tensor size of [height, width, volume] of input
        image.
      num_classes: The number of classes to be segmented.
      num_channels: The channel of input images.
      image_field_key: A `str` of the key name to encoded image in TFExample.
      label_field_key: A `str` of the key name to label in TFExample.
      dtype: The data type. One of {`bfloat16`, `float32`, `float16`}.
      label_dtype: The data type of input label.
    """
    self._input_size = input_size
    self._num_classes = num_classes
    self._num_channels = num_channels
    self._image_field_key = image_field_key
    self._label_field_key = label_field_key
    self._image_shape_key = image_shape_key
    self._label_shape_key = label_shape_key
    self._dtype = dtype
    self._label_dtype = label_dtype

  def _prepare_image_and_label_tr(
      self, data: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares normalized image and label."""
    image = data[self._image_field_key]
    if isinstance(image, tf.SparseTensor):
      image = tf.sparse.to_dense(image)

    label = data[self._label_field_key]
    if isinstance(label, tf.SparseTensor):
      label = tf.sparse.to_dense(label)

    image_size = data[self._image_shape_key]
    image = tf.reshape(image, image_size)

    label_size = data[self._label_shape_key]
    label = tf.reshape(label, label_size)
    label = tf.cast(label, dtype=tf.float32)

    image, label = self._data_augmentation_tr(image, label)

    image.set_shape(self._input_size+[self._num_channels])
    label.set_shape(self._input_size+[self._num_classes])

    image = tf.cast(image, dtype=self._dtype)
    label = tf.cast(label, dtype=self._label_dtype)

    # TPU doesn't support tf.int64 well, use tf.int32 directly.
    if label.dtype == tf.int64:
      label = tf.cast(label, dtype=tf.int32)
    return image, label

  def _prepare_image_and_label_val(
      self, data: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares normalized image and label."""
    image = data[self._image_field_key]
    if isinstance(image, tf.SparseTensor):
      image = tf.sparse.to_dense(image)

    label = data[self._label_field_key]
    if isinstance(label, tf.SparseTensor):
      label = tf.sparse.to_dense(label)

    image_size = data[self._image_shape_key]
    image = tf.reshape(image, image_size)

    label_size = data[self._label_shape_key]
    label = tf.reshape(label, label_size)
    label = tf.cast(label, dtype=tf.float32)

    image, label = self._data_augmentation_val(image, label)

    image.set_shape(self._input_size+[self._num_channels])
    label.set_shape(self._input_size+[self._num_classes])

    image = tf.cast(image, dtype=self._dtype)
    label = tf.cast(label, dtype=self._label_dtype)

    # TPU doesn't support tf.int64 well, use tf.int32 directly.
    if label.dtype == tf.int64:
      label = tf.cast(label, dtype=tf.int32)
    return image, label

  def _data_augmentation_tr(self, image, label):
    image, label = process_batch(image, label[tf.newaxis,], [73, 80, 64], [40, 56, 40])
    image, label = tf_tr_transforms(image, label)
    return image[0], label[0]

  def _data_augmentation_val(self, image, label):
    image, label = process_batch(image, label[tf.newaxis,], [40, 56, 40], [40, 56, 40])
    image, label = tf_val_transforms(image, label)
    return image[0], label[0]

  def _parse_train_data(self, data: Dict[str,
                                         Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for training and evaluation."""
    image, labels = self._prepare_image_and_label_tr(data)
    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels

  def _parse_eval_data(self, data: Dict[str,
                                        Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parses data for training and evaluation."""
    image, labels = self._prepare_image_and_label_val(data)
    # Cast image as self._dtype
    image = tf.cast(image, dtype=self._dtype)

    return image, labels


def tr_transforms(image, label):
    data_dict = {'data': image, 'seg': label}
    data_dict = SpatialTransform(patch_size=[40, 56, 40], patch_center_dist_from_border=None,
                                 do_elastic_deform=False, alpha=(0.0, 900.0), sigma=(9.0, 13.0),
                                 do_rotation=True, angle_x=(-0.5235987755982988, 0.5235987755982988),
                                 angle_y=(-0.5235987755982988, 0.5235987755982988),
                                 angle_z=(-0.5235987755982988, 0.5235987755982988), p_rot_per_axis=1,
                                 do_scale=True, scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=0,
                                 order_data=3, border_mode_seg="constant", border_cval_seg=-1, order_seg=1,
                                 random_crop=False, p_el_per_sample=0.2, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
                                 independent_scale_for_each_axis=False)(**data_dict)
    data_dict = GaussianNoiseTransform(p_per_sample=0.1)(**data_dict)
    data_dict = GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                      p_per_channel=0.5)(**data_dict)
    data_dict = BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15)(**data_dict)
    data_dict = ContrastAugmentationTransform(p_per_sample=0.15)(**data_dict)
    data_dict = SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True, p_per_channel=0.5,
                                               order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                               ignore_axes=None)(**data_dict)
    data_dict = GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1)(**data_dict)
    data_dict = GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3)(**data_dict)
    data_dict = MirrorTransform((0, 1, 2))(**data_dict)
    data_dict = MaskTransform([(0, False)], mask_idx_in_seg=0, set_outside_to=0)(**data_dict)
    data_dict = RemoveLabelTransform(-1, 0)(**data_dict)

    image = data_dict['data']
    label = data_dict['seg']
    return image, label


def val_transforms(image, label):
    data_dict = {'data': image, 'seg': label}

    data_dict = RemoveLabelTransform(-1, 0)(**data_dict)

    image = data_dict['data']
    label = data_dict['seg']
    return image, label


# @tf.function
def process_batch(
    image,
    label,
    basic_generator_patch_size,
    patch_size,
):
    zero = tf.constant(0, dtype=tf.int64)
    image = tf.cast(image, dtype=tf.float32)
    label = tf.cast(label, dtype=tf.float32)

    case_all_data = tf.concat([image, label], axis=0)
    basic_generator_patch_size = tf.cast(
        basic_generator_patch_size, dtype=tf.int64
    )
    patch_size = tf.cast(patch_size, dtype=tf.int64)
    need_to_pad = basic_generator_patch_size - patch_size
    need_to_pad = tf.map_fn(
        lambda d: update_need_to_pad(
            need_to_pad, d, basic_generator_patch_size, case_all_data
        ),
        elems=tf.range(3, dtype=tf.int64),
    )
    need_to_pad = tf.cast(need_to_pad, tf.int64)
    shape = tf.shape(case_all_data, out_type=tf.int64)[1:]
    lb_x = -need_to_pad[0] // 2
    ub_x = (
        shape[0]
        + need_to_pad[0] // 2
        + need_to_pad[0] % 2
        - basic_generator_patch_size[0]
    )
    lb_y = -need_to_pad[1] // 2
    ub_y = (
        shape[1]
        + need_to_pad[1] // 2
        + need_to_pad[1] % 2
        - basic_generator_patch_size[1]
    )
    lb_z = -need_to_pad[2] // 2
    ub_z = (
        shape[2]
        + need_to_pad[2] // 2
        + need_to_pad[2] % 2
        - basic_generator_patch_size[2]
    )

    bbox_x_lb, bbox_y_lb, bbox_z_lb = not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z)

    bbox_x_ub = bbox_x_lb + basic_generator_patch_size[0]
    bbox_y_ub = bbox_y_lb + basic_generator_patch_size[1]
    bbox_z_ub = bbox_z_lb + basic_generator_patch_size[2]

    valid_bbox_x_lb = tf.maximum(zero, bbox_x_lb)
    valid_bbox_x_ub = tf.minimum(shape[0], bbox_x_ub)
    valid_bbox_y_lb = tf.maximum(zero, bbox_y_lb)
    valid_bbox_y_ub = tf.minimum(shape[1], bbox_y_ub)
    valid_bbox_z_lb = tf.maximum(zero, bbox_z_lb)
    valid_bbox_z_ub = tf.minimum(shape[2], bbox_z_ub)

    case_all_data = tf.identity(
        case_all_data[
            :,
            valid_bbox_x_lb:valid_bbox_x_ub,
            valid_bbox_y_lb:valid_bbox_y_ub,
            valid_bbox_z_lb:valid_bbox_z_ub,
        ]
    )

    img = tf.pad(
        case_all_data[:-1],
        (
            [0, 0],
            [
                -tf.minimum(zero, bbox_x_lb),
                tf.maximum(bbox_x_ub - shape[0], zero),
            ],
            [
                -tf.minimum(zero, bbox_y_lb),
                tf.maximum(bbox_y_ub - shape[1], zero),
            ],
            [
                -tf.minimum(zero, bbox_z_lb),
                tf.maximum(bbox_z_ub - shape[2], zero),
            ],
        ),
        mode="CONSTANT",
    )
    seg = tf.pad(
        case_all_data[-1:],
        (
            [0, 0],
            [
                -tf.minimum(zero, bbox_x_lb),
                tf.maximum(bbox_x_ub - shape[0], zero),
            ],
            [
                -tf.minimum(zero, bbox_y_lb),
                tf.maximum(bbox_y_ub - shape[1], zero),
            ],
            [
                -tf.minimum(zero, bbox_z_lb),
                tf.maximum(bbox_z_ub - shape[2], zero),
            ],
        ),
        mode="CONSTANT",
        constant_values=-1,
    )
    return img[tf.newaxis,], seg[tf.newaxis,]

def update_need_to_pad(
    need_to_pad, d, basic_generator_patch_size, case_all_data
):
    need_to_pad_d = (
        basic_generator_patch_size[d]
        - tf.shape(case_all_data, out_type=tf.int64)[d + 1]
    )
    return tf.cond(
        tf.less(
            need_to_pad[d] + tf.shape(case_all_data, out_type=tf.int64)[d + 1],
            basic_generator_patch_size[d],
        ),
        lambda: need_to_pad_d,
        lambda: need_to_pad[d],
    )

def not_force_fg(lb_x, ub_x, lb_y, ub_y, lb_z, ub_z):
    bbox_x_lb = tf.random.uniform(
        [], minval=lb_x, maxval=ub_x + 1, dtype=tf.int64
    )
    bbox_y_lb = tf.random.uniform(
        [], minval=lb_y, maxval=ub_y + 1, dtype=tf.int64
    )
    bbox_z_lb = tf.random.uniform(
        [], minval=lb_z, maxval=ub_z + 1, dtype=tf.int64
    )
    return bbox_x_lb, bbox_y_lb, bbox_z_lb


def tf_tr_transforms(images, segs):
  # tf.config.run_functions_eagerly(True)
  data_dict = TFDAData(data=images, seg=segs)
  # tf.print(tf.shape(data_dict.data), tf.shape(data_dict.seg))

  data_dict = SpatialTransform(
    patch_size=[40, 56, 40], #
    patch_center_dist_from_border=nan,
    do_elastic_deform=TFbF, # 
    alpha=(0.0, 900.0), #
    sigma=(9.0, 13.0), #
    do_rotation=True, 
    # angle_x=(tf.constant(-0.5235987755982988), tf.constant(0.5235987755982988)), #
    angle_x=(-0.5235987755982988, 0.5235987755982988), #
    angle_y=(tf.constant(-0.5235987755982988), tf.constant(0.5235987755982988)), #
    angle_z=(tf.constant(-0.5235987755982988), tf.constant(0.5235987755982988)), #
    p_rot_per_axis=1.0, #
    do_scale=True, #
    scale=(0.7, 1.4), #
    border_mode_data='constant', 
    border_cval_data=0,
    order_data=3, 
    border_mode_seg="constant", 
    border_cval_seg=-1, 
    order_seg=1,
    random_crop=False, #
    p_el_per_sample=0.2, #
    p_scale_per_sample=0.2, #
    p_rot_per_sample=0.2, #
    independent_scale_for_each_axis=False #
    )(data_dict)
    # tf.config.run_functions_eagerly(False)

  data_dict = GaussianNoiseTransform(
    data_key="data", label_key="seg", p_per_channel=0.01)(data_dict)
  
  data_dict = GaussianBlurTransform(
    (0.5, 1.), 
    different_sigma_per_channel=True, 
    p_per_sample=0.2,
    p_per_channel=0.5
    )(data_dict)
  
  data_dict = BrightnessMultiplicativeTransform(
    multiplier_range=(0.75, 1.25), 
    p_per_sample=0.15
    )(data_dict)
  
  data_dict = ContrastAugmentationTransform(p_per_sample=0.15)(data_dict)
  
  data_dict = SimulateLowResolutionTransform(
    zoom_range=(0.5, 1), 
    per_channel=True, 
    p_per_channel=0.5,
    order_downsample=0, 
    order_upsample=3, 
    p_per_sample=0.25
    # ignore_axes=None
    )(data_dict)
    
  data_dict = GammaTransform(
    (0.7, 1.5), #
    True, True, 
    retain_stats=True, #
    p_per_sample=0.1
    )(data_dict)
  
  data_dict = GammaTransform(
    (0.7, 1.5), #
    False, True, 
    retain_stats=True, #
    p_per_sample=0.3 #
    )(data_dict)
  
  data_dict = MirrorTransform(
    (0, 1, 2) #
    )(data_dict) 
  
  data_dict = MaskTransform(
    tf.constant([[0, 0], [1, 0]]), #
    mask_idx_in_seg=0, 
    set_outside_to=0.0
    )(data_dict)

  data_dict = RemoveLabelTransform(-1, 0)(data_dict)
  
  data_dict = OneHotTransform(
    tuple([float(key) for key in [0, 1, 2]]) #
    )(data_dict) 

  # tf.print('tr', tf.shape(data_dict.data))
  return data_dict.data, data_dict.seg

def tf_val_transforms(images, segs):
  data_dict = TFDAData(data=images, seg=segs)
  data_dict = RemoveLabelTransform(-1, 0)(TFDAData(data=images, seg=segs))
  data_dict = OneHotTransform(
    tuple([float(key) for key in [0, 1, 2]])
    )(data_dict) #
  # tf.print('val', tf.shape(data_dict.data))
  return data_dict.data, data_dict.seg
