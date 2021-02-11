import numpy as np
from imagegrab import grab_screen_rgb
import cv2
import os
import time
from keys import PressKey, ReleaseKey, keyA, keyD
from getkeys import key_check


WIDTH = 300
HEIGHT = 200
w = [1, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0]
a = [0, 0, 1, 0, 0]
d = [0, 0, 0, 1, 0]
n = [0, 0, 0, 0, 1]

starting_value = 1

while True:
    file_name = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{}.npy'.format(starting_value)

    if os.path.isfile(file_name):
        print('File exists, moving along', starting_value)
        starting_value += 1
    else:
        print('File does not exist, starting fresh!', starting_value)
        break


def keys_to_output(keys):
    output = [0, 0, 0, 0, 0]  # [W, S, A, D, NoKey]

    if 'W' in keys and 'A' in keys:
        output = a
    elif 'W' in keys and 'D' in keys:
        output = d
    elif 'S' in keys and 'A' in keys:
        output = s
    elif 'S' in keys and 'D' in keys:
        output = s
    elif 'W' in keys:
        output = w
    elif 'S' in keys:
        output = s
    elif 'A' in keys:
        output = a
    elif 'D' in keys:
        output = d
    else:
        output = n

    return output


def main(file_name, starting_value):
    file_name = file_name
    starting_value = starting_value
    training_data = []
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    paused = False
    print('STARTING!!!')
    while True:

        if not paused:
            screen = grab_screen_rgb(640, 34, 1920, 834)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))

            keys = key_check()
            output = keys_to_output(keys)
            training_data.append([screen, output])

            if len(training_data) == 500:
                np.save(file_name, training_data)
                print(f'saved {file_name}')
                training_data = []
                starting_value += 1
                file_name = 'D:/Ayudesee/Other/Data/ets-data-raw-rgb/training_data-{}.npy'.format(starting_value)

        keys = key_check()
        if 'T' in keys:
            if paused:
                paused = False
                print('Unpaused!')
                time.sleep(1)
            else:
                print('Pausing!')
                paused = True
                time.sleep(1)


main(file_name, starting_value)
