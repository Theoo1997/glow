#!/bin/bash

export glow=../../../../build_Debug_llvm11/bin
BASEDIR="$PWD/"

#Quantization Parameters
precision=Int8
bias=Int32
schema=asymmetric

echo "_________________image-classifier _____________________________"
$glow/image-classifier "$BASEDIR"dataset/perf_samples/*.png  -image-channel-order=BGR -model="$BASEDIR"pretrainedResnet.tflite -model-input-name=input_1 -image-layout=NHWC -dump-profile="$BASEDIR"dataset/profile.yml -quantization-schema=$schema -quantization-precision=$precision -quantization-precision-bias=$bias

echo "_________________image-tuner _____________________________"
$glow/model-tuner -backend=CPU -model="$BASEDIR"pretrainedResnet.tflite -image-channel-order=BGR -image-mode=0to255 -dataset-path="$BASEDIR"dataset/perf_samples -dataset-file="$BASEDIR"dataset/y_labels.csv -load-profile="$BASEDIR"dataset/profile.yml -model-input=input_1,float,[1,32,32,3] -image-layout=NHWC -dump-tuned-profile="$BASEDIR"profile_tuner.yml -quantization-schema=$schema -quantization-precision=$precision -quantization-precision-bias=$bias

