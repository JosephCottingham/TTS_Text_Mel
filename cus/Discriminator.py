import tensorflow as tf
from tensorflow.keras import layers

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(300, name='Dense0', input_shape=(870, 80)))
    model.add(layers.Dense(300, name='Dense1'))
    model.add(layers.Dropout(0.3, name='Dropout1'))

    model.add(layers.Dense(300, name='Dense2'))
    model.add(layers.Dropout(0.3, name='Dropout2'))

    model.add(layers.Dense(261, name='Dense3'))
    model.add(layers.Dropout(0.3, name='Dropout3'))

    model.add(layers.Dense(1, name='Dense4'))

    return model