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

fullsum = np.full(5, 0)

for _i in range(1, FILE_I_END):
    df = pd.read_csv(f'D:/Ayudesee/Other/Data/ets-data-choices/choices-{_i}.csv')
    df.drop(columns="Unnamed: 0", inplace=True)

    fullsum[0] += (df["w"] == 1).sum()
    fullsum[1] += (df["s"] == 1).sum()
    fullsum[2] += (df["a"] == 1).sum()
    fullsum[3] += (df["d"] == 1).sum()
    fullsum[4] += (df["n"] == 1).sum()

print(fullsum)

fullsum.reshape(5, 1)
df2 = pd.DataFrame([fullsum], columns=["w", "s", "a", "d", "n"])

print(df2)
