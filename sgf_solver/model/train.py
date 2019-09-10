import os

from .model import create_model


def train_model(inputs, outputs):
    model = create_model()
    model.summary()

    if os.path.exists('weights.h5'):
        model.load_weights('weights.h5')

    return inputs, outputs
