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
        self.model_speed = tf.keras.models.load_model('model_num_v3.h5')
        for file in glob.glob(f'{path_to_data}/*.npy'):
            self.files.append(file)
        random.shuffle(self.files)

    def __len__(self):
        return math.ceil(len(self.files) * 500 / self.batch_size)

    def __getitem__(self, idx):
        speed = []
        if idx % (500 / self.batch_size) == 0:
            data = np.load(self.files[int(idx // (500 / self.batch_size))], allow_pickle=True)
            self.data = data

        batch_x = self.data[int(idx % (500 / self.batch_size)): int((idx % (500 / self.batch_size)) + self.batch_size), 0]
        batch_y = self.data[int(idx % (500 / self.batch_size)): int((idx % (500 / self.batch_size)) + self.batch_size), 1]
        for i in range(len(batch_x)):
            speed.append(self.model_speed.predict(np.expand_dims(batch_x[i][0:18, 14:34, :], axis=0)))

        speed = np.reshape(np.array(speed), (self.batch_size, 1))

        # imgs1, imgs2, imgs3 = process_images(batch_x)
        # batch_x = np.stack([imgs1, imgs2, imgs3], axis=1)
        batch_x = process_images(batch_x)
        # batch_x = np.array([np.reshape(_, (200, 300, 3)) for _ in batch_x])
        batch_y = np.array([np.reshape(_, 9) for _ in batch_y])
        # batch_x = np.insert(batch_x, speed, axis=1)
        # cv2.imshow(f'{idx}', batch_x[0])
        # cv2.waitKey(0)
        # time.sleep(1)
        return {"image1": batch_x[:, 0, :, :, :], "image2": batch_x[:, 1, :, :, :], "image3": batch_x[:, 2, :, :, :], "speed": speed}, batch_y

    def on_epoch_end(self):
        random.shuffle(self.files)


# path_to_dataset = 'D:/Ayudesee/Other/Data/ets-data-shuffled-balanced'
# my_gen = MyCustomGenerator(path_to_dataset, 10)
#
# print(my_gen[0])