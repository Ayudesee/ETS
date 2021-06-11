import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, ConvLSTM2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import elu


def alexnet_model_modified(img_shape=(200, 300, 3), n_classes=9, weights=None):
    alexnet = Sequential()

    alexnet.add(Conv2D(64, (7, 7), input_shape=img_shape, padding='same'))
    alexnet.add(Conv2D(96, (7, 7), input_shape=img_shape, padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))

    alexnet.add(Conv2D(128, (5, 5), padding='same'))
    alexnet.add(Conv2D(128, (5, 5), padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))

    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))

    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))

    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))

    alexnet.add(Flatten())
    alexnet.add(Dense(1024))
    alexnet.add(Dropout(0.3))
    alexnet.add(Activation('tanh'))

    alexnet.add(Dense(256))
    alexnet.add(Activation('tanh'))

    alexnet.add(Dense(n_classes))
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    alexnet.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    alexnet.summary()
    return alexnet


def alexnet_model_modified_v2(img_shape=(200, 300, 3), n_classes=9, weights=None):
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), input_shape=img_shape, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    #

    # Layer 2
    alexnet.add(Conv2D(filters=256, kernel_size=(5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3)))

    # Layer 3
    alexnet.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Layer 4
    alexnet.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(4096))
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(4096))
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))
    alexnet.add(Dense(n_classes))
    alexnet.add(Activation('softmax'))

    alexnet.compile(optimizer=Adam(learning_rate=0.1), loss='categorical_crossentropy', metrics=['accuracy'])
    alexnet.summary()
    return alexnet


def alexnet_model_modified_v3(img_shape=(200, 300, 3), n_classes=10, l2_reg=0., weights=None):
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(32, (11, 11), input_shape=img_shape,
                       padding='same', kernel_regularizer=l2(l2_reg)))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    alexnet.add(Dropout(0.3))

    # Layer 2
    alexnet.add(Conv2D(64, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(Conv2D(128, (3, 3), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Layer 4
    # alexnet.add(ZeroPadding2D((1, 1)))
    # alexnet.add(Conv2D(512, (3, 3), padding='same'))
    # alexnet.add(BatchNormalization())
    # alexnet.add(Activation('relu'))
    #
    # # Layer 5
    # alexnet.add(ZeroPadding2D((1, 1)))
    # alexnet.add(Conv2D(1024, (3, 3), padding='same'))
    # alexnet.add(BatchNormalization())
    # alexnet.add(Activation('relu'))
    # alexnet.add(MaxPooling2D(pool_size=(2, 2)))


    # Layer 6

    alexnet.add(Flatten())
    alexnet.add(Dense(256))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(256))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    alexnet.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    alexnet.summary()
    return alexnet
