#!/bin/bash
# Downoad the dataset.
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xvf cifar-10-python.tar.gz

# Preprocess the dataset and and strore it under images.
python3 perf_samples_loader.py