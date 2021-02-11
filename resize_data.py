import numpy as np
import cv2

FILE_I_END = 720

for _ in range(1, FILE_I_END + 1):
    file1 = np.load(f"D:/Ayudesee/Other/Data/ets-data-shuffled/training_data-{_}.npy", allow_pickle=True)
    print(_)
    for data in file1:
        data[0] = cv2.resize(data[0], (108, 68))
    np.save(f"D:/Ayudesee/Other/Data/ets-data-shuffled-108-68/training_data-{_}.npy", file1)
