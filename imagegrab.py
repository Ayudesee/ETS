from PIL import ImageGrab
import numpy as np
import cv2


def grab_screen(start_x, start_y, width, height):
    screen = np.array(ImageGrab.grab(bbox=(start_x, start_y, width, height)))  # 0, 32, 768, 512 sh
    return screen


def grab_screen_rgb(start_x, start_y, width, height):
    screen = np.array(ImageGrab.grab(bbox=(start_x, start_y, width, height)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    return screen


# while True:
#     window = grab_screen_rgb(640, 34, 1920, 834)
#     window = cv2.resize(window, (432, 270))
#     cv2.imshow('window', window)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
