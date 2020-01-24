import numpy as np

from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense
from tensorflow.keras.applications.vgg16 import preprocess_input


def vgg16_no_head(num_classes=2, verbose=True, freeze=True):
    model_vgg16_conv = VGG16(weights='imagenet', include_top=False)

    if freeze:
        # Freezes VGG layers
        for layer in model_vgg16_conv.layers:
            layer.trainable = False

    # Pass input through VGG
    vgg_input = Input(shape=(200, 200, 3), name="ImageInput")
    vgg_output = model_vgg16_conv(vgg_input)

    # Add fully-connected layers
    x = Flatten(name="conv_flatten")(vgg_output)
    x = Dense(1024, activation="relu", name="fc1")(x)
    x = Dense(256, activation="relu", name="fc2")(x)
    x = Dense(num_classes, activation="softmax", name="prediction")(x)

    # DeepPap model
    model = Model(vgg_input, x)
    if verbose: model.summary()

    return model