from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam


def xception(img_shape=(200, 300, 3), n_classes=9):
    base_model = Xception(weights=None, include_top=False, input_shape=img_shape)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    opt = Adam(learning_rate=0.001, decay=1e-5)

    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
