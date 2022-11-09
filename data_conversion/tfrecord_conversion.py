import os
import pickle
import argparse

import numpy as np
import tensorflow as tf

# tfrecord feature keys
IMAGE_KEY = 'image/encoded'
CLASSIFICATION_LABEL_KEY = 'image/class/label'
IMAGE_SHAPE_KEY = 'image_shape'
LABEL_SHAPE_KEY = 'label_shape'


def convert_one_sample(data_folder, fold, file_name, save_path):
    data = np.load(os.path.join(data_folder, (file_name + ".npz")))['data']
    image = data[:-1, :, :, :]
    label = data[-1, :, :, :]
    label = label.astype(np.int64)
    feature = {
        IMAGE_KEY: (tf.train.Feature(
            float_list=tf.train.FloatList(value=image.flatten()))),
        CLASSIFICATION_LABEL_KEY: (tf.train.Feature(
            int64_list=tf.train.Int64List(value=label.flatten()))),
        IMAGE_SHAPE_KEY: (tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(image.shape)))),
        LABEL_SHAPE_KEY: (tf.train.Feature(
            int64_list=tf.train.Int64List(value=list(label.shape))))
    }
    tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
    tfrecord_file = os.path.join(save_path, ('fold{}'.format(fold) + '_val_' + file_name + '.tfrecord'))

    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        writer.write(tf_example.SerializeToString())


def main(splits_file, fold, data_path, save_path):
    with open(splits_file, 'rb') as f:
        splits = pickle.load(f)
    val_keys = splits[fold]['val']
    val_keys.sort()

    for i, file_name in enumerate(val_keys):
        print("converting fold", fold, "validation case:", i)
        convert_one_sample(data_path, fold, file_name, save_path)
    print("done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configurations for data format conversion.')
    parser.add_argument('--data_path', type=str, default='/mnt/SSD1/fengtong/nnunet/nnUNet_preprocessed', 
                        help='nnUNet preprocessed data.')
    parser.add_argument('--task', type=int, default=4, 
                        help='Task id.')
    parser.add_argument('--network', type=str, choices=['2d','3d'], default='3d',
                        help='Network architecture.')
    args = parser.parse_args()

    nnUNet_preprocessed = args.data_path
    task_id = args.task
    network_architecture = args.network

    task_name = {
        1: "Task001_BrainTumour",
        2: "Task002_Heart",
        3: "Task003_Liver",
        4: "Task004_Hippocampus",
        5: "Task005_Prostate",
        6: "Task006_Lung",
        7: "Task007_Pancreas",
      }  
    preprocessed_task_path = os.path.join(nnUNet_preprocessed, task_name[task_id])

    # 5-fold cross-validation. Fold: 0, 1, 2, 3, 4
    for fold in range(5):
        if network_architecture == '3d':
            if task_id in [1, 2, 4, 5]:
                data_folder = os.path.join(preprocessed_task_path, "nnUNetData_plans_v2.1_stage0")  # 3d: task 001, 002, 004, 005
            else:
                data_folder = os.path.join(preprocessed_task_path, "nnUNetData_plans_v2.1_stage1")  # 3d: task 003, 006, 007
        else:
            data_folder = os.path.join(preprocessed_task_path, "nnUNetData_plans_v2.1_2D_stage0")  # 2d: all tasks

        splits_file = os.path.join(preprocessed_task_path, "splits_final.pkl")
        save_path = os.path.join(preprocessed_task_path, (network_architecture + "_tfrecord_data"))

        if not os.path.exists(save_path):
            os.mkdir(save_path)

        main(splits_file, fold, data_folder, save_path)