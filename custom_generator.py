import math
import time

import tensorflow as tf
import numpy as np
import glob
import random
from process_image import process_images
import cv2


class MyCustomGenerator(tf.keras.utils.Sequence):

    def __init__(self, path_to_data, batch_size):
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.files = []
        for file in glob.glob(f'{path_to_data}/*.npy'):
            self.files.append(file)
        random.shuffle(self.files)

    def __len__(self):
        return math.ceil(len(self.files) * 500 / self.batch_size)

    def __getitem__(self, idx):
        if idx % (500 / self.batch_size) == 0:
            data = np.load(self.files[int(idx // (500 / self.batch_size))], allow_pickle=True)
            self.data = data

        batch_x = self.data[int(idx % (500 / self.batch_size)): int((idx % (500 / self.batch_size)) + self.batch_size), 0]
        batch_y = self.data[int(idx % (500 / self.batch_size)): int((idx % (500 / self.batch_size)) + self.batch_size), 1]

        # batch_x = process_images(batch_x)
        batch_x = np.array([np.reshape(_, (200, 300, 3)) for _ in batch_x])
        batch_y = np.array([np.reshape(_, 9) for _ in batch_y])

        # cv2.imshow(f'{idx}', batch_x[0])
        # cv2.waitKey(0)
        # time.sleep(1)
        return batch_x, batch_y

    def on_epoch_end(self):
        random.shuffle(self.files)
