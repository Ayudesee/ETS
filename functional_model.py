import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, Input, concatenate


def func_model(input_shape=(2, 3, 200, 300, 3), n_classes=9):
    visible_1 = Input(shape=(200, 300, 3), name='image1')
    conv_1_1 = Conv2D(8, (5, 5), strides=(3, 3), padding='same', activation='relu')(visible_1)
    conv_1_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_1_1)
    maxp_1_1 = MaxPooling2D((2, 2))(conv_1_2)

    visible_2 = Input(shape=(200, 300, 3), name='image2')
    conv_2_1 = Conv2D(8, (5, 5), strides=(3, 3), padding='same', activation='relu')(visible_2)
    conv_2_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_2_1)
    maxp_2_1 = MaxPooling2D((2, 2))(conv_2_2)

    visible_3 = Input(shape=(200, 300, 3), name='image3')
    conv_3_1 = Conv2D(16, (3, 3), strides=(3, 3), padding='same', activation='relu')(visible_3)
    conv_3_2 = Conv2D(32, (3, 3), padding='same', activation='relu')(conv_3_1)
    maxp_3_1 = MaxPooling2D((2, 2))(conv_3_2)

    visible_4 = Input(shape=1, name='speed')
    flatten_1 = Flatten()(maxp_1_1)
    flatten_2 = Flatten()(maxp_2_1)
    flatten_3 = Flatten()(maxp_3_1)

    conc_layer_1 = concatenate([flatten_1, flatten_2, flatten_3])
    dense1 = Dense(256, activation='relu')(conc_layer_1)
    dense2 = Dense(128, activation='relu')(dense1)
    conc_layer_2 = concatenate([dense2, visible_4])
    dense3 = Dense(32, activation='relu')(conc_layer_2)
    output = Dense(n_classes, activation='softmax')(dense3)

    model = Model(inputs=[visible_1, visible_2, visible_3, visible_4], outputs=output)

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics='accuracy')
    model.summary()
    tf.keras.utils.plot_model(model, "func_v1.png")
    return model
