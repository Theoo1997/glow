#!/bin/bash

export PATH_compiler=../../../../build_Debug_llvm11/bin
 
$PATH_compiler/model-compiler -backend=CPU -emit-bundle=./bundle --bundle-api=static -model=person_detect_quant.tflite #-llvm-compiler="llvm_install/bin/clang"
