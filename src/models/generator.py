
import tensorflow as tf
import numpy as np


class Post_Matching(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.input_layer = tf.keras.layers.Dense(
            self.config['n_mels'],
            name='Dense_Input',
            activation='linear'
        )
        self.dense1 = tf.keras.layers.SimpleRNN(
            3000,
            name='SimpleRNN_1',
            activation='relu'
        )  
        self.dense2 = tf.keras.layers.SimpleRNN(
            3000,
            name='SimpleRNN_2',
            activation='relu'
        )
        self.output_layer = tf.keras.layers.Dense(
            self.config['n_mels'],
            name='Dense_Output',
            activation='linear'
        )   

class Post_Matching_Dense(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.input_layer = tf.keras.layers.Dense(
            self.config['n_mels'],
            name='Dense_Input',
            activation='linear'
        )
        self.dense1 = tf.keras.layers.Dense(
            3000,
            name='Dense_1',
            activation='relu'
        )  
        self.dense2 = tf.keras.layers.Dense(
            3000,
            name='Dense_2',
            activation='relu'
        )
        self.output_layer = tf.keras.layers.Dense(
            self.config['n_mels'],
            name='Dense_Output',
            activation='linear'
        )    


    def call(self, inputs, training=False):
        """Call logic."""
        output = self.input_layer(inputs)
        output = self.dense1(inputs)
        output = self.dense2(inputs)
        output = self.output_layer(inputs)
        return output


class Generator(tf.keras.Model):
    
    base_model = None

    def __init__(self, config, base_model, **kwargs):
        super().__init__(self, **kwargs)

        self.config = config
        self.base_model = base_model
        
        self.post_matching = Post_Matching(
            self.config
        )

    
    def call(
        self,
        inputs,
        training=False,
        **kwargs,
    ):
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = self.base_model(**inputs, training=training, **kwargs)

        post_matching_outputs = self.post_matching(mel_outputs)

        mel_outputs += post_matching_outputs

        return (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        )

    def _build(self):
        inputs = {}
        inputs['input_ids'] = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
        inputs['input_lengths'] = np.array([9])
        inputs['speaker_ids'] = np.array([0])
        inputs['mel_gts'] = np.random.normal(size=(1, 50, 80)).astype(np.float32)
        inputs['mel_lengths'] = np.array([50])
        self(
            inputs,
            training=False,
        )

    def inference(self, input_ids, input_lengths, speaker_ids):
        inputs = {}
        inputs['input_ids'] = input_ids
        inputs['input_lengths'] = input_lengths
        inputs['speaker_ids'] = speaker_ids
        inputs['mel_gts'] = np.random.normal(size=(1, 50, 80)).astype(np.float32)
        inputs['mel_lengths'] = np.array([50])
        return self(
            inputs,
            training=False,
        )