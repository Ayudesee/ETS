import numpy as np
import cv2


end_value = 186

for _ in range(end_value):
    filename_bgr = 'D:/Ayudesee/Other/Data/ets-data-raw/training_data-{}.npy'.format(_ + 1)
    filename_rgb = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{}.npy'.format(_ + 1)

    data = np.load(filename_bgr, allow_pickle=True)
    for _ in range(500):
        data[_][0] = cv2.cvtColor(data[_][0], cv2.COLOR_BGR2RGB)

    np.save(filename_rgb, data)
