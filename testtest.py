from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split


file_name = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-1.npy'
train_data = np.load(file_name, allow_pickle=True)

images = []
keys = []

for data in train_data:
    images.append(data[0])
    keys.append(data[1])

X, x_test, Y, y_test = train_test_split(images, keys, test_size=0.1)
print(len(Y), len(y_test))