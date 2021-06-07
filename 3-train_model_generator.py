import tensorflow as tf
from alexnet import alexnet_model_modified
from custom_generator import MyCustomGenerator

path_to_dataset = 'D:/Ayudesee/Other/Data/ets-data-shuffled-balanced'
WIDTH = 300
HEIGHT = 200
MODEL_NAME = '07-06-2ranges-v1-generator'
logdir = f".\\logs\\{MODEL_NAME}"
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir, profile_batch=5)

LOAD_MODEL = False

model = alexnet_model_modified(img_shape=(HEIGHT, WIDTH, 3), n_classes=9)

if LOAD_MODEL:
    model.load(MODEL_NAME)
    print('We have loaded a previous model!!!!')


my_gen = MyCustomGenerator(path_to_dataset, 10)

model.fit(my_gen, verbose=1, epochs=10, callbacks=tensorboard_callback)
print('SAVING MODEL!')
model.save(f"models/{MODEL_NAME}.h5")


#tensorboard --logdir=D:/Ayudesee/Other/PyProj/ETS/logs