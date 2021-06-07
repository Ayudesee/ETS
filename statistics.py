import pandas as pd
import numpy as np
import os


FILE_I_END = 0
while True:
    file_name = 'D:/Ayudesee/Other/Data/ets-data-choices/choices-{}.csv'.format(FILE_I_END + 1)

    if os.path.isfile(file_name):
        FILE_I_END += 1
    else:
        print(f'files found: {FILE_I_END}')
        break

fullsum = np.full(9, 0)

for _i in range(1, FILE_I_END + 1):
    df = pd.read_csv(f'D:/Ayudesee/Other/Data/ets-data-choices/choices-{_i}.csv')
    df.drop(columns="Unnamed: 0", inplace=True)

    fullsum[0] += (df["w"] == 1).sum()
    fullsum[1] += (df["s"] == 1).sum()
    fullsum[2] += (df["a"] == 1).sum()
    fullsum[3] += (df["d"] == 1).sum()
    fullsum[4] += (df["wa"] == 1).sum()
    fullsum[5] += (df["wd"] == 1).sum()
    fullsum[6] += (df["sa"] == 1).sum()
    fullsum[7] += (df["sd"] == 1).sum()
    fullsum[8] += (df["nk"] == 1).sum()

print(fullsum)

fullsum.reshape(9, 1)
df2 = pd.DataFrame([fullsum], columns=["w", "s", "a", "d", "wa", "wd", "sa", "sd", "n"])

print(df2)
