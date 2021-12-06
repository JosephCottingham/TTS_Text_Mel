
import tensorflow as tf


class Post_Matching(tf.keras.layers.Layer):

    def __init__(self, config):
        self.config = config
        self.dense = tf.keras.layers.Dense(
            1000
        )
        self.dropout = tf.keras.layers.Dropout(
            rate=.01, name="dropout_1"
        )


class Generator(tf.keras.Model):

    base_model = None

    def __init__(self, config, base_model):
        super().__init__(self, **kwargs)

        self.config = config
        self.base_model = base_model
        self.post_matching = Post_Matching(
            self.config['Post_Matching']
        )
    
    def call(
        self,
        inputs,
        training=False
        **kwargs,
    ):
        outputs = self.base_model(**inputs, training=training, **kwargs)
        outputs = self.post_matching(outputs)

        return outputs