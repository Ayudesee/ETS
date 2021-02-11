import tensorflow as tf
import time
import os
from random import random
from process_image import process_image
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
LR = 1e-3
EPOCHS = 10

# MODEL_NAME = 'model1'


def main():
    filepath = 'model_wrapped_1-682'
    paused = False

    model = tf.keras.models.load_model(filepath=filepath)
    for i in list(range(5))[::-1]:
        print(i + 1)
        time.sleep(1)

    while True:
        if not paused:
            screen = grab_screen_rgb(640, 34, 1920, 834)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            screen = process_image(screen)
            cv2.imshow('window', screen)
            cv2.waitKey(1)

            screen = np.reshape(screen, (-1, HEIGHT, WIDTH, 3))
            prediction = model.predict(screen)

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
                time.sleep(0.02)
                ReleaseKey(A)
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
                time.sleep(0.02)
                ReleaseKey(D)
            elif mode_choice == 4:
                no_keys()
                choice_picked = 'no_keys'

            np.set_printoptions(formatter={'float': '{: 0.2f}'.format})
            print(choice_picked, prediction)

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
