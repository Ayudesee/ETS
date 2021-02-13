import pandas as pd
import numpy as np
import os

FILE_I_END = 1

while True:
    file_name = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{}.npy'.format(FILE_I_END)

    if os.path.isfile(file_name):
        FILE_I_END += 1
    else:
        print('files found:', FILE_I_END - 1)
        break

for _i in range(1, FILE_I_END):
    choices = []
    file = np.load(f'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{_i}.npy', allow_pickle=True)
    print(f'processing... ({_i}/{FILE_I_END-1})')
    for _k in range(0, 500):
        choices.append(file[_k][1])

    choices = np.array(choices)
    choices.reshape(9, -1)
    df = pd.DataFrame(choices, columns=["w", "s", "a", "d", "wa", "wd", "sa", "sd", "nk"])
    df.to_csv(f'D:/Ayudesee/Other/Data/ets-data-choices/choices-{_i}.csv')
