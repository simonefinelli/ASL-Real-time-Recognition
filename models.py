"""
The proposed models are valid for the WLASL20custom, WLASL100, WLASL300,
WLASL1000 and WLASL2000 datasets.
"""
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.layers import TimeDistributed, GRU, BatchNormalization
from tensorflow.keras.models import Sequential


def create_model_wlasl20c(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),

        # GRUs
        GRU(256, return_sequences=True),
        BatchNormalization(),
        GRU(256),

        # Feedforward
        Dense(units=64, activation='relu'),
        Dropout(0.65),
        Dense(units=32, activation='relu'),
        Dropout(0.65),
        Dense(units=output, activation='softmax')
    ])

    return model


def create_model_wlasl100(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),

        # GRUs
        GRU(256, return_sequences=True),
        BatchNormalization(),
        GRU(256),

        # Feedforward
        Dense(units=200, activation='relu'),
        Dropout(0.66),
        Dense(units=150, activation='relu'),
        Dropout(0.66),
        Dense(units=output, activation='softmax')
    ])

    return model


def create_model_wlasl300(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),

        # GRUs
        GRU(256, return_sequences=True),
        BatchNormalization(),
        GRU(256),

        # Feedforward
        Dense(units=400, activation='relu'),
        Dropout(0.6),
        Dense(units=350, activation='relu'),
        Dropout(0.6),
        Dense(units=output, activation='softmax')
    ])

    return model


def create_model_wlasl1000(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),

        # GRUs
        GRU(256, return_sequences=True),
        BatchNormalization(),
        GRU(256),

        # Feedforward
        Dense(units=512, activation='relu'),
        Dropout(0.7),
        Dense(units=output, activation='softmax')
    ])

    return model


def create_model_wlasl2000(frames, width, height, channels, output):
    """
    Create the keras model.

    :param frames: frame number of the sequence.
    :param width: width of the image.
    :param height: height of the image.
    :param channels: 3 for RGB, 1 for B/W images.
    :param output: number of neurons for classification.
    :return: the keras model.
    """
    model = Sequential([
        # ConvNet
        TimeDistributed(
            MobileNetV2(weights='imagenet', include_top=False,
                        input_shape=[height, width, channels]),
            input_shape=[frames, height, width, channels]
        ),
        TimeDistributed(GlobalAveragePooling2D()),

        # GRUs
        GRU(256, return_sequences=True),
        BatchNormalization(),
        GRU(256),

        # Feedforward
        Dense(units=1000, activation='relu'),
        Dropout(0.75),
        Dense(units=output, activation='softmax')
    ])

    return model
