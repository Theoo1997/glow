# Quantization of MLPerf Tiny image classification reference models on Glow

This are the MLPerf Tiny image classification reference models.

ResNet8,MobilenetV1 models are trained on the CIFAR10, VWW datasets correspondingly.

Model: ResNet8 / MobilenetV1
Dataset: Cifar10 / VWW

## Quick start

Run the following commands to go through the whole training and validation process

``` Bash
# Download the dataset and preprosses it
cd $model_name/dataset
./download_and_make_dataset.sh

# Quantize the model usind model Tuner
cd ..
./quantize.sh
```