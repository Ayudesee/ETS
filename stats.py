import pandas as pd
import numpy as np
import os

FILE_I_END = 1

while True:
    file_name = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{}.npy'.format(FILE_I_END)

    if os.path.isfile(file_name):
        print('File exists, moving along', FILE_I_END)
        FILE_I_END += 1
    else:
        print('File does not exist, starting fresh!', FILE_I_END)
        break

for _i in range(682, FILE_I_END):
    choices = []
    file = np.load(f'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{_i}.npy', allow_pickle=True)
    print(_i)
    for _k in range(0, 500):
        choices.append(file[_k][1])

    choices = np.array(choices)
    choices.reshape(5, -1)
    df = pd.DataFrame(choices, columns=["w", "s", "a", "d", "n"])
    df.to_csv(f'D:/Ayudesee/Other/Data/ets-data-choices/choices-{_i}.csv')
