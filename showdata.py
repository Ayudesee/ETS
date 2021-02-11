import numpy as np
import cv2

testpack1 = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-50.npy'

image1 = np.load(testpack1, allow_pickle=True)

n = 180
cv2.imshow(str(n), image1[n][0])
print(image1[n][1])
if cv2.waitKey(0) & 0xFF == ord('q'):
    cv2.destroyAllWindows()

