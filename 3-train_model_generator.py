import tensorflow as tf
from alexnet import alexnet_model_modified, alexnet_model_modified_v2
from vgg_16 import vgg_16_modified
from xception import xception
from custom_generator import MyCustomGenerator
import datetime


path_to_dataset = 'D:/Ayudesee/Other/Data/ets-data-shuffled-balanced'
path_to_dataset_val = 'D:/Ayudesee/Other/Data/ets-data-shuffled-balanced_val'
WIDTH = 300
HEIGHT = 200

MODEL_NAME = f'{datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")}'
logdir = f".\\logs\\{MODEL_NAME}"

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=5)

LOAD_MODEL = False

# model = alexnet_model_modified(img_shape=(HEIGHT, WIDTH, 3), n_classes=9)
# model = vgg_16_modified(img_shape=(HEIGHT, WIDTH, 3), n_classes=9)
model = xception(img_shape=(HEIGHT, WIDTH, 3), n_classes=9)

if LOAD_MODEL:
    model.load(MODEL_NAME)
    print('We have loaded a previous model!!!!')
print(f'modelname: {MODEL_NAME}')

my_gen = MyCustomGenerator(path_to_dataset, 10)
my_gen_val = MyCustomGenerator(path_to_dataset_val, 10)

model.fit(my_gen, verbose=1, epochs=5, callbacks=tensorboard_callback, validation_data=my_gen_val, shuffle=False)
print('SAVING MODEL!')
model.save(f"models/{MODEL_NAME}.h5")


#tensorboard --logdir=D:/Ayudesee/Other/PyProj/ETS/logs