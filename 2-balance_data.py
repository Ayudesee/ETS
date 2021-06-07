import numpy as np
import os
from random import shuffle


# FILE_START = 1
FILE_I_END = 1

w = [1, 0, 0, 0, 0, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0, 0, 0, 0, 0]
a = [0, 0, 1, 0, 0, 0, 0, 0, 0]
d = [0, 0, 0, 1, 0, 0, 0, 0, 0]
wa = [0, 0, 0, 0, 1, 0, 0, 0, 0]
wd = [0, 0, 0, 0, 0, 1, 0, 0, 0]
sa = [0, 0, 0, 0, 0, 0, 1, 0, 0]
sd = [0, 0, 0, 0, 0, 0, 0, 1, 0]
nk = [0, 0, 0, 0, 0, 0, 0, 0, 1]

while True:
    file_name = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{}.npy'.format(FILE_I_END)

    if os.path.isfile(file_name):
        FILE_I_END += 1
    else:
        print(f'files found: {FILE_I_END - 1}')
        break

final_data = []
counter = 0
for _ in range(1, FILE_I_END):
    cut_off_fwd = 18
    cut_off_nk = 14
    train_data = np.load(f'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{_}.npy', allow_pickle=True)

    forwards = []
    nokeys = []
    other = []

    for data in train_data:
        if data[1] == w:
            forwards.append([data[0], data[1]])
        elif data[1] == nk:
            nokeys.append([data[0], data[1]])
        else:
            other.append([data[0], data[1]])

    shuffle(forwards)
    shuffle(nokeys)

    forwards = forwards[0:int(len(forwards)/cut_off_fwd)]
    nokeys = nokeys[0:int(len(nokeys)/cut_off_nk)]

    final_data.extend(other)
    final_data.extend(forwards)
    final_data.extend(nokeys)
    shuffle(final_data)
    print(_)
    if len(final_data) >= 500:
        counter += 1
        np.save(f'D:/Ayudesee/Other/Data/ets-data-shuffled-balanced/training_data-{counter}.npy', final_data[:500])
        final_data = final_data[500:]
