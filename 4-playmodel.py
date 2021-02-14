import tensorflow as tf
import time
import os
from random import random
from process_image import process_image_v5
from imagegrab import grab_screen_rgb
import numpy as np
import cv2
from directkeys import ReleaseKey, PressKey, W, S, A, D, straight, left, right, reverse, forward_right, forward_left, \
    reverse_left, reverse_right, no_keys
from getkeys import key_check

vertices = np.array([[0, 8], [51, 8], [54, 0], [91, 0], [95, 11], [151, 11], [151, 103], [0, 103]])  # main screen coords
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
WIDTH = 300
HEIGHT = 200
LR = 1e-2
EPOCHS = 10

# MODEL_NAME = 'model1'


def main():
    filepath = 'models/model_proc_img_v4_5_0.01.h5'
    paused = False

    model = tf.keras.models.load_model(filepath=filepath)
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    while True:
        # print(time.time())
        if not paused:
            screen = grab_screen_rgb(640, 34, 1920, 834)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            screen = process_image_v5(screen)
            cv2.imshow('window', screen)
            cv2.waitKey(1)

            screen = np.reshape(screen, (-1, HEIGHT, WIDTH, 3))
            prediction = model.predict(screen) * np.array([0.7, 0.9, 0.25, 0.25, 0.27, 0.27, 2, 2, 0.4])

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
                # time.sleep(0.095)
                # ReleaseKey(A)
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
                # time.sleep(0.095)
                # ReleaseKey(D)
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
                time.sleep(0.07)
                ReleaseKey(A)
                ReleaseKey(W)
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
                time.sleep(0.07)
                ReleaseKey(D)
                ReleaseKey(W)
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            else:
                no_keys()
                choice_picked = 'no_keys'

            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(prediction, choice_picked)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('unpaused!')
                time.sleep(1)
            else:
                paused = True
                print('Pausing!')
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                ReleaseKey(S)
                time.sleep(1)


main()
