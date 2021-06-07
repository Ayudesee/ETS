import math
import tensorflow as tf
import numpy as np
import glob
from process_image import process_images


class MyCustomGenerator(tf.keras.utils.Sequence):

    def __init__(self, path_to_data, batch_size):
        self.path_to_data = path_to_data
        self.batch_size = batch_size
        self.files = []
        for file in glob.glob(f'{path_to_data}/*.npy'):
            self.files.append(file)

    def __len__(self):
        return math.ceil(len(self.files) / self.batch_size)


    def __getitem__(self, idx):
        if idx * self.batch_size % 500 == 0:
            data = np.load(self.files[int(idx // (500 / self.batch_size))], allow_pickle=True)
            self.data = data

        batch_x = self.data[idx * self.batch_size: (idx + 1) * self.batch_size, 0]
        batch_y = self.data[idx * self.batch_size: (idx + 1) * self.batch_size, 1]
        #batch_x = process_images(batch_x)

        # for _ in range(len(batch_x)):
        #     batch_x = np.reshape(batch_x[_], (-1, 200, 300, 3))

        batch_x = np.array([np.reshape(_, (200, 300, 3)) for _ in batch_x])
        batch_y = np.array([np.reshape(_, 9) for _ in batch_y])
        return batch_x, batch_y
