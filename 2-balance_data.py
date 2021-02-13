import numpy as np
import os
from random import shuffle


FILE_START = 1
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


for _ in range(FILE_START, FILE_I_END):
    cut_off_fwd = 9
    cut_off_nk = 9
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

    # print(f'{_} - len(fwd)/{cut_off_fwd} = {int(len(forwards)/cut_off_fwd)}')

    forwards = forwards[0:int(len(forwards)/cut_off_fwd)]
    nokeys = nokeys[0:int(len(nokeys)/cut_off_nk)]

    final_data = other + nokeys + forwards
    shuffle(final_data)

    print(f'{_} - total data length:{len(final_data)}, fwd:{int(100 * int(len(forwards))/len(final_data))}%, nk:{int(100 * int(len(nokeys))/len(final_data))}%')
    np.save(f'D:/Ayudesee/Other/Data/ets-data-shuffled-9-9-cutoff/training_data-{_}.npy', final_data)
