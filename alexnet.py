"""
AlexNet Keras Implementation
BibTeX Citation:
@inproceedings{krizhevsky2012imagenet,
  title={Imagenet classification with deep convolutional neural networks},
  author={Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
  booktitle={Advances in neural information processing systems},
  pages={1097--1105},
  year={2012}
}
"""

# Import necessary packages
import argparse

# Import necessary components to build LeNet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization


def alexnet_model_modified(img_shape=(200, 300, 3), n_classes=9, weights=None):
    alexnet = Sequential()

    # Layer 1
    alexnet.add(Conv2D(64, (15, 15), input_shape=img_shape, padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 2
    alexnet.add(Conv2D(96, (7, 7), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))

    # Layer 3
    alexnet.add(Conv2D(128, (5, 5), padding='same'))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(MaxPooling2D(pool_size=(2, 2)))
    #
    # # Layer 4
    # alexnet.add(Conv2D(256, (3, 3), padding='same'))
    # alexnet.add(BatchNormalization())
    # alexnet.add(Activation('relu'))

    # Layer 6
    alexnet.add(Flatten())
    alexnet.add(Dense(512))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.5))

    # Layer 7
    alexnet.add(Dense(256))
    alexnet.add(BatchNormalization())
    alexnet.add(Activation('relu'))
    alexnet.add(Dropout(0.3))

    # Layer 8
    alexnet.add(Dense(n_classes))
    alexnet.add(BatchNormalization())
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
