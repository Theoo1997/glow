import numpy as np
import os
import train
import struct
from PIL import Image
import matplotlib.image
import cv2


perf_samples_dir = 'perf_samples'
cifar_10_dir = 'cifar-10-batches-py'

if __name__ == '__main__':
    if not os.path.exists(perf_samples_dir):
        os.makedirs(perf_samples_dir)

    label_output_file = open('y_labels.csv', 'w')

    _idxs = np.load('perf_samples_idxs.npy')
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)

    for i in _idxs:
        _output_str = '{name},{label},\n'.format(name=test_filenames[i].decode('UTF-8')[:-3] + 'png', classes=10, label=np.argmax(test_labels[i]))
        label_output_file.write(_output_str)
        sample_img = np.array(test_data[i])
        sample_img = sample_img.astype('float32')
        pixels = np.asarray(sample_img)
        if len(pixels.shape)== 3:
            cv2.imwrite("./perf_samples/" + test_filenames[i].decode('UTF-8'), sample_img)
    label_output_file.close()
