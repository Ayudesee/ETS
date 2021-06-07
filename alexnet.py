import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.activations import elu


def alexnet_model_modified(img_shape=(200, 300, 3), n_classes=9, weights=None):
    alexnet = Sequential()

    alexnet.add(Conv2D(64, (7, 7), input_shape=img_shape, padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))
    alexnet.add(Conv2D(128, (5, 5), padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))
    alexnet.add(Conv2D(256, (3, 3), padding='same'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    alexnet.add(Activation('elu'))
    alexnet.add(Flatten())
    alexnet.add(Dense(64))
    alexnet.add(Activation('elu'))
    alexnet.add(Dropout(0.3))
    alexnet.add(Dense(n_classes))
    alexnet.add(Activation('softmax'))

    if weights is not None:
        alexnet.load_weights(weights)

    alexnet.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
    alexnet.summary()
    return alexnet


def parse_args():
    """
	Parse command line arguments.
	Parameters:
		None
	Returns:
		parser arguments
	"""
    parser = argparse.ArgumentParser(description='AlexNet model')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional.add_argument('--print_model',
                          dest='print_model',
                          help='Print AlexNet model',
                          action='store_true')
    parser._action_groups.append(optional)
    return parser.parse_args()


if __name__ == "__main__":
    # Command line parameters
    args = parse_args()

    # Create AlexNet model
    model = alexnet_model_modified()

    # Print
    if args.print_model:
        model.summary()
