import numpy as np
import os
from random import shuffle


FILE_START = 1
FILE_I_END = 1

while True:
    file_name = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{}.npy'.format(FILE_I_END)

    if os.path.isfile(file_name):
        print('File exists, moving along', FILE_I_END)
        FILE_I_END += 1
    else:
        print('File does not exist, starting fresh!', FILE_I_END)
        break


for _ in range(FILE_START, FILE_I_END):
    train_data = np.load(f'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{_}.npy', allow_pickle=True)
    forwards = []
    nokeys = []
    other = []

    for data in train_data:
        if data[1] == [1, 0, 0, 0, 0]:
            forwards.append([data[0], data[1]])
        elif data[1] == [0, 0, 0, 0, 1]:
            nokeys.append([data[0], data[1]])
        else:
            other.append([data[0], data[1]])

    shuffle(forwards)
    shuffle(nokeys)

    cut_off_fwd = 9
    cut_off_nk = 7

    print(f'{_} - len(fwd)/{cut_off_fwd} = {int(len(forwards)/cut_off_fwd)}')

    forwards = forwards[0:int(len(forwards)/cut_off_fwd)]
    nokeys = nokeys[0:int(len(nokeys)/cut_off_nk)]

    final_data = other + nokeys + forwards
    shuffle(final_data)
    np.save(f'D:/Ayudesee/Other/Data/ets-data-shuffled-9-7-cutoff/training_data-{_}.npy', final_data)
