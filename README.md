# TensorFlow nnUNet 
This TF nnUNet code is implemented based on TensorFlow volumetric models from TensorFlow Model Garden vision projects. The TensorFlow Model Garden is used as a subfolder in `tf_nnunet/models`. After downloading this repository, the `PYTHONPATH` should be set to `XXX(path to tf_nnunet folder)/tf_nnunet/models`.

## nnUNet Installation

Install nnUNet by following the instructions on [nnUNet codebase](https://github.com/MIC-DKFZ/nnUNet). Set up environment variables to let nnUNet know where raw data, preprocessed data and trained model weights are stored.

## Download Dataset
Download Medical Segmentation Decathlon datasets from http://medicaldecathlon.com. 

## Dataset Conversion
Convert the datasets into the correct format by following [dataset conversion instructions](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md). The converted dataset can be found in `$nnUNet_raw_data_base/nnUNet_raw_data`.

## Data Preprocessing
Run `nnUNet_plan_and_preprocess` by following the [experiment planning and preprocessing](https://github.com/MIC-DKFZ/nnUNet#experiment-planning-and-preprocessing) section. It will preprocess rew data in `$nnUNet_raw_data_base/nnUNet_raw_data/TaskXXX_MYTASK` and populate preprocessed data in `$nnUNet_preprocessed/TaskXXX_MYTASK`.

## TFRecord Conversion
`data_conversion/tfrecord_conversion.py` converts the preprocessed data to tfrecord files, so that the data format is compatible with TF Vision. The dataset is split to 5 folds according to a splits file located in `$nnUNet_preprocessed/TaskXXX_MYTASK/splits_final.pkl`. A fold number is added as a prefix to the name of every data sample. For example, after tfrecord conversion, a npz data sample `hippocampus_001.npz` is converted to tfrecord format file `fold0_val_hippocampus_001.tfrecord`. The prefix, `fold0_val`, means when we do 5-fold cross validation, this sample belongs to fold 0. We can use samples with other prefixes as training set, and all the samples with prefix `fold0_val` as validation set. 

## Data Augmentations
In the current version, all the parameters for data augmentations are given in `dataloaders/segmentation_input_3d_msd.py`. The customized dataloader file includes the data augmentation methods and corresponding parameters for the 3d architecture model of all tasks. All the tensorflow data augmentations are implemented in the `data_augmentations` folder.

## Train the Model 
The experiment configurations can be found in yaml files under the experiments folder. To train the model, you can run the following lines with overriding some parameters in yaml file:

```bash
python /…/models/official/projects/volumetric_models/train.py \
--experiment=seg_unet3d_test \
--mode=train_and_eval \
--config_file=YAML_FILE_ PATH \
--model_dir=OUTPUT_PATH \
--params_override="task.train_data.input_path=[ \
nnUNet_preprocessed/Task004_Hippocampus/3d_tfrecord_data/fold1*, \
nnUNet_preprocessed/Task004_Hippocampus/3d_tfrecord_data/fold2*, \
nnUNet_preprocessed/Task004_Hippocampus/3d_tfrecord_data/fold3*, \
nnUNet_preprocessed/Task004_Hippocampus/3d_tfrecord_data/fold4*], \
task.validation_data.input_path= \
nnUNet_preprocessed/Task004_Hippocampus/3d_tfrecord_data/fold0*, \
trainer.checkpoint_interval=2500, \
trainer.validation_interval=2500, \
trainer.steps_per_loop=2500, \
trainer.summary_interval=2500, \
trainer.train_steps=25000"
```

## Export Trained Model
`serving/export_saved_model.py` exports a trained checkpoint so the model can be used in inference later. Following lines show the parameters of the command:

```bash
python3 /…/models/official/projects/volumetric_models/serving/export_saved_model.py \
--experiment=seg_unet3d_test \
--export_dir=/…/exported_model \
--checkpoint_path==SAVED_CKPT_PATH \
--config_file=YAML_FILE_PATH \
--batch_size=1 \
--input_image_size=40,56,40 \
--num_channels=1
```

## Inference and Postprocessing
Replace the inference model in nnUNet code with exported TF model. nnUNet will automatically do the postprocessing and report the final results of 5-fold CV.



