from random import shuffle
import numpy as np
from sklearn.model_selection import train_test_split


file_name = 'D:/Ayudesee/Other/Data/ets-data-shuffled/training_data-1.npy'
train_data = np.load(file_name, allow_pickle=True)

X, Y, x_test, y_test = train_test_split(train_data[0], train_data[1], test_size=0.1)

print(Y)
# print(Y)