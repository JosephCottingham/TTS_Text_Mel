import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv1D(64, (4,), strides=(2), padding='same', input_shape=(32, 80), name='Conv1D1'))
    model.add(layers.Dense(3000, name='Dense1'))
    model.add(layers.Dropout(0.3, name='Dropout1'))

    model.add(layers.Conv1D(128, (4,), strides=(2), padding='same', name='Conv1D2'))
    model.add(layers.Dense(3000, name='Dense2'))
    model.add(layers.Dropout(0.3, name='Dropout2'))

    model.add(layers.Flatten(name='Flatten1'))

    model.add(layers.Dense(261, name='Dense3'))
    model.add(layers.Dropout(0.3, name='Dropout3'))

    model.add(layers.Dense(1, name='Dense4'))

    return model