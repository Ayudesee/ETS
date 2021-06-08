import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from process_image import process_images
from alexnet import alexnet_model_modified, alexnet_model_modified_v2
from random import shuffle
import matplotlib.pyplot as plt


print(tf.config.get_visible_devices())
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.threading.set_inter_op_parallelism_threads = 1
# tf.config.threading.set_intra_op_parallelism_threads = 1

# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         # Memory growth must be set before GPUs have been initialized
#         print(e)
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         tf.config.experimental.set_virtual_device_configuration(gpus[0],
#                                                                 [tf.config.experimental.VirtualDeviceConfiguration(
#                                                                     memory_limit=3072)])
#         logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#     except RuntimeError as e:
#         print(e)

FILE_I_END = 1
path = 'D:/Ayudesee/Other/Data/ets-data-shuffled-balanced/training_data-{}.npy'

while True:
    file_name = path.format(FILE_I_END)
    if os.path.isfile(file_name):
        FILE_I_END += 1
    else:
        print('FILE_I_END = ', FILE_I_END)
        break

WIDTH = 300
HEIGHT = 200
LR = 1e-3
EPOCHS = 20

MODEL_NAME = '07-06-test-2'  # .format(EPOCHS, LR)
PREV_MODEL = ''
logdir = f".\\logs\\{MODEL_NAME}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=5)

LOAD_MODEL = False

model = alexnet_model_modified_v2(img_shape=(HEIGHT, WIDTH, 3), n_classes=9)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')

for e in range(EPOCHS):
    print(f"--------------_______________***************_____________________EPOCH:{e+1}___________________***************_________________--------------")
    data_order = [i for i in range(1, FILE_I_END)]  #   data_order = [i for i in range(1, FILE_I_END)]
    shuffle(data_order)
    for count, i in enumerate(data_order):
        try:
            file_name = path.format(i)
            train_data = np.load(file_name, allow_pickle=True)
            print(f'training_data-{i}.npy, _EPOCH:{e+1}/{EPOCHS}, ({count+1}/{FILE_I_END-1})')

            images = []
            keys = []
            for data in train_data:
                images.append(data[0])
                keys.append(data[1])

            X, test_x, Y, test_y = train_test_split(images, keys, test_size=0.01)

            X = np.array(X).reshape(-1, HEIGHT, WIDTH, 3)
            Y = np.array(Y)
            # test_x = np.array(test_x).reshape(-1, HEIGHT, WIDTH, 3)
            # test_y = np.array(test_y)

            X = process_images(X)
            # test_x = process_images(test_x)

            model.fit(X, Y, epochs=1, batch_size=10, verbose=1, callbacks=tensorboard_callback)#  validation_data=(test_x, test_y))  # , validation_freq=10, use_multiprocessing=True, callbacks=[tensorboard_callback])
            # model.train_on_batch(X, Y)
            # model.fit_generator((X, Y), steps_per_epoch= FILE_I_END-1)

            if count % 30 == 0 or count == FILE_I_END-1:
                print('SAVING MODEL!')
                model.save(f"models/{MODEL_NAME}.h5")

        except Exception as err:
            print(str(err))
#tensorboard --logdir=D:/Ayudesee/Other/PyProj/ETS/logs