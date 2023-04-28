#!/bin/bash

export glow=../../../../build_Debug_llvm11/bin
BASEDIR="$PWD/"

$glow/image-classifier "$BASEDIR"dataset/images/*.png  -image-channel-order=RGB -image-mode=0to1 -model="$BASEDIR"vww_96.tflite -model-input-name=input_1 -image-layout=NHWC -dump-profile="$BASEDIR"dataset/profile.yml

$glow/model-tuner -backend=CPU -model="$BASEDIR"vww_96.tflite -image-channel-order=RGB -image-mode=0to1 -dataset-path="$BASEDIR"dataset/images/ -dataset-file="$BASEDIR"dataset/y_labels.txt -load-profile="$BASEDIR"dataset/profile.yml -model-input=input_1,float,[1,96,96,3] -image-layout=NHWC -dump-tuned-profile="$BASEDIR"profile_tuner.yml #-device=0 -num-devices=1 -opencl-profile -opencl-specialize-convolution
