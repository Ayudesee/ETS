import numpy as np
import tensorflow as tf
import os
from process_image import process_images
from alexnet import alexnet_model_modified
from random import shuffle

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# tf.config.threading.set_inter_op_parallelism_threads = 1
# tf.config.threading.set_intra_op_parallelism_threads = 1

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)
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

while True:
    file_name = 'D:/Ayudesee/Other/Data/ets-data-shuffled-9-7-cutoff/training_data-{}.npy'.format(FILE_I_END)
    if os.path.isfile(file_name):
        FILE_I_END += 1
    else:
        print('FILE_I_END = ', FILE_I_END)
        break

WIDTH = 300
HEIGHT = 200
LR = 1e-3
EPOCHS = 3

MODEL_NAME = 'model_proc_img_v3'
PREV_MODEL = ''
logdir = f"logs/{MODEL_NAME}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

LOAD_MODEL = False

w = [1, 0, 0, 0, 0]
s = [0, 1, 0, 0, 0]
a = [0, 0, 1, 0, 0]
d = [0, 0, 0, 1, 0]
n = [0, 0, 0, 0, 1]

model = alexnet_model_modified(img_shape=(HEIGHT, WIDTH, 3), n_classes=5)

if LOAD_MODEL:
    model.load(PREV_MODEL)
    print('We have loaded a previous model!!!!')

for e in range(EPOCHS):
    print(f"--------------_______________***************_____________________EPOCH:{e+1}___________________***************_________________--------------")
    data_order = [i for i in range(1, FILE_I_END)]  #   data_order = [i for i in range(1, FILE_I_END)]
    shuffle(data_order)
    for count, i in enumerate(data_order):
        try:
            file_name = 'D:/Ayudesee/Other/Data/ets-data-shuffled-9-7-cutoff/training_data-{}.npy'.format(i)
            train_data = np.load(file_name, allow_pickle=True)
            print(f'training_data-{i}.npy, _EPOCH:{e+1}, ({count+1}/{FILE_I_END-1})')

            train = train_data[:-10]
            test = train_data[-10:]

            X = np.array([i[0] for i in train]).reshape(-1, HEIGHT, WIDTH, 3)
            Y = np.array([i[1] for i in train])
            X = process_images(X)

            test_x = np.array([i[0] for i in test]).reshape(-1, HEIGHT, WIDTH, 3)
            test_y = np.array([i[1] for i in test])
            test_x = process_images(test_x)

            model.fit(X, Y, epochs=1, batch_size=10, validation_data=(test_x, test_y), verbose=0)# , callbacks=tensorboard_callback)

            if count % 30 == 0 or count == FILE_I_END-1:
                print('SAVING MODEL!')
                model.save(f"models/{MODEL_NAME}.h5")

        except Exception as err:
            print(str(err))

#tensorboard --logdir=D:/Ayudesee/Other/PyProj/ETS/logs