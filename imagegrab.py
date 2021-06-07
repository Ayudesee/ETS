from PIL import ImageGrab
import numpy as np
import cv2


def grab_screen(start_x, start_y, end_x, end_y):
    screen = np.array(ImageGrab.grab(bbox=(start_x, start_y, end_x, end_y)))  # 0, 32, 768, 512 sh
    return screen


def grab_screen_rgb(start_x, start_y, end_x, end_y):
    screen = np.array(ImageGrab.grab(bbox=(start_x, start_y, end_x, end_y)))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    return screen


# while True:
#     window = grab_screen_rgb(640, 34, 1920, 834)
#     window_speed = grab_screen_rgb(1594, 558, 1670, 576)
#
#     window = cv2.resize(window, (432, 270))
#     window[:18, :76, :] = window_speed
#     cv2.imshow('window', window)
#     cv2.imshow('window_speed', window_speed)
#     if cv2.waitKey(10) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
