#!/bin/bash
# Downoad the dataset.
wget https://www.silabs.com/public/files/github/machine_learning/benchmarks/datasets/vw_coco2014_96.tar.gz
tar -xvf vw_coco2014_96.tar.gz

# Preprocess the dataset and and strore it under images.
python3 make_dataset.py