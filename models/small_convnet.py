import tensorflow as tf


def small_convnet(image_shape=(128, 128, 3), num_classes=2, verbose=True,
                  dropout_rate=0.0, normalize=False):
    input_layer = tf.keras.layers.Input(shape=image_shape)
    # Conv + pool
    conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding="same")(input_layer)
    if normalize: conv3 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.ReLU()(conv1)
    max_pool1 = tf.keras.layers.MaxPooling2D(strides=(2, 2), padding="same")(conv1)

    # Conv + pool
    conv2 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, padding="same")(max_pool1)
    if normalize: conv3 = tf.keras.layers.BatchNormalization()(conv2)
    conv2 = tf.keras.layers.ReLU()(conv2)
    max_pool2 = tf.keras.layers.MaxPooling2D(strides=(2, 2), padding="same")(conv2)

    # Conv + pool
    conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=7, strides=(2, 2), padding="same")(max_pool2)
    if normalize: conv3 = tf.keras.layers.BatchNormalization()(conv3)
    conv3 = tf.keras.layers.ReLU()(conv3)
    max_pool3 = tf.keras.layers.MaxPooling2D(strides=(2, 2), padding="same")(conv3)

    # Final conv
    conv4 = tf.keras.layers.Conv2D(filters=16, kernel_size=7, strides=(2, 2), padding="same")(max_pool3)
    if normalize: conv3 = tf.keras.layers.BatchNormalization()(conv4)
    conv4 = tf.keras.layers.ReLU()(conv4)

    # Classifier
    flat = tf.keras.layers.Flatten()(conv4)
    fc1 = tf.keras.layers.Dense(1024, activation='relu')(flat)
    if dropout_rate > 0.0: fc1 = tf.keras.layers.Dropout(rate=dropout_rate)(fc1)
    fc2 = tf.keras.layers.Dense(256, activation='relu')(fc1)
    if dropout_rate > 0.0: fc2 = tf.keras.layers.Dropout(rate=dropout_rate)(fc2)
    fc3 = tf.keras.layers.Dense(num_classes, activation='softmax')(fc2)

    return tf.keras.models.Model(input_layer, fc3)