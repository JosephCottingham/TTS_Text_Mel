import os
import json

import tensorflow as tf
import numpy as np

from tensorflow_tts.utils import utils
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy

class Trainer():

    def __init__(
        self,
        config,
        STRATEGY,
        steps=0,
        epochs=0
    ):

        self.config = config
        self.STRATEGY = STRATEGY
        self.saved_path = self.config["outdir"] + "/checkpoints/"
        os.makedirs(self.saved_path, exist_ok=True)

        self.steps = steps
        self.epochs = epochs

        self.list_metrics_name = [
            "adversarial_loss",
            "fm_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
            "mels_spectrogram_loss",
        ]

        
    def _train_step(self, batch):

        # print('_train_step')
        # Key batch data 'input_lengths', 'speaker_ids', 'mel_gts', 'mel_lengths'
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = self._generator.inference(
            tf.convert_to_tensor(batch['input_ids'], dtype=tf.int32),
            tf.convert_to_tensor(batch['input_lengths'], tf.int32),
            tf.convert_to_tensor(batch['speaker_ids'], dtype=tf.int32)
        )        # outputs from generator

        mel_loss_before = calculate_3d_loss(
            batch["mel_gts"], decoder_output, loss_fn=self.mae_loss
        )
        mel_loss_after = calculate_3d_loss(
            batch["mel_gts"], mel_outputs, loss_fn=self.mae_loss
        )


        # Padd or Slice generator output so that the discrimator can accept the shape
        mel_outputs_shape = mel_outputs.get_shape()
        add = self.config["max_mel_length"] - mel_outputs_shape[1]
        # Checks if padding/slicing is required
        if add < 0:
            mel_outputs = tf.slice(mel_outputs, [0,0,0], [mel_outputs_shape[0], self.config["max_mel_length"], mel_outputs_shape[2]])
        else:
            paddings = tf.constant([[0, 0], [0, add], [0,0]])
            mel_outputs = tf.pad(mel_outputs, paddings, "CONSTANT")

        batch_real_loss = 0.0
        batch_fake_loss = 0.0
        # Gather predictions
        for step in range(32):
            with tf.GradientTape() as gtape:
            
                p_hat = self._discriminator(mel_outputs[step])
                p = self._discriminator(batch['mel_gts'][step])

                real_loss = self.mse_loss(tf.zeros_like(p), p)
                fake_loss = self.mse_loss(tf.ones_like(p), p_hat)
                
                batch_fake_loss += tf.math.reduce_mean(real_loss).numpy()
                batch_real_loss += tf.math.reduce_mean(real_loss).numpy()

                gtape.watch(self._discriminator.trainable_variables)
                gtape.watch(fake_loss)

                gradients = gtape.gradient(
                    fake_loss, self._discriminator.trainable_variables
                )

                self._dis_optimizer.apply_gradients(
                    zip(gradients, self._discriminator.trainable_variables)
                )

                gradients = gtape.gradient(
                    real_loss, self._discriminator.trainable_variables
                )

                self._dis_optimizer.apply_gradients(
                    zip(gradients, self._discriminator.trainable_variables)
                )

        batch_real_loss /= (step + 1)
        batch_fake_loss /= (step + 1)


        # adv_loss = 0.0
        # fm_loss = 0.0
        # for i in range(len(p_hat)):
        #     adv_loss += calculate_2d_loss(
        #         tf.ones_like(p_hat[i]), p_hat[i], loss_fn=self.mae_loss
        #     )
        #     fm_loss += calculate_2d_loss(
        #         p[i], p_hat[i], loss_fn=self.mae_loss
        #     )
        # fm_loss /= (i + 1)
        # adv_loss /= i + 1

        # real_loss = []
        # fake_loss = []
        # for i in range(len(p)):
        #     real_loss.append(calculate_3d_loss(
        #         tf.ones_like(p[i]), p[i], loss_fn=self.mse_loss
        #     ))
        #     fake_loss.append(calculate_3d_loss(
        #         tf.zeros_like(p_hat[i]), p_hat[i], loss_fn=self.mse_loss
        #     ))
        # # real_loss /= i + 1
        # # fake_loss /= i + 1
        # # dis_loss = real_loss + fake_loss
        # print(f'{ self.steps }', end=' - ')

        self.dict_metrics_losses = {
            "real_loss": batch_real_loss,
            "fake_loss": batch_fake_loss,
            # "adversarial_loss": adv_loss,
            # "fm_loss": fm_loss,
            # "gen_loss": adv_loss
        }

        # self.steps += 1



        # # Generator
        # # per_replica_gen_losses = tf.nn.compute_average_loss(
        # #     adv_loss,
        # #     global_batch_size=self.config["batch_size"]
        # # )


        # # self._gen_optimizer.apply_gradients(
        # #     zip(gradients, self._generator.trainable_variables)
        # # )
        # # https://www.tensorflow.org/tutorials/customization/custom_training_walkthrough

        # per_replica_dis_losses = tf.nn.compute_average_loss(
        #     real_loss,
        #     global_batch_size=self.config["batch_size"]
        # )
        # # with STRATEGY.scope():
        # #     loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
        # #     l = loss_object(y_true=p, y_pred=p_hat)


        # with tf.GradientTape() as gtape:
        #     for step_index in range(len(p_hat)):
        #         l = self.mse_loss(tf.ones_like(p[i]), p_hat[i])
        #         print(l)
        #         print('-------')
        #         gradients = gtape.gradient(
        #             l, self._generator.trainable_variables
        #         )
        #         print(gradients)
        #         self._dis_optimizer.apply_gradients(
        #             zip(gradients, self._discriminator.trainable_variables)
        #         )

        # per_replica_losses = per_replica_gen_losses + per_replica_dis_losses
        # STRATEGY.reduce(
        #     tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None
        # )

        return self.dict_metrics_losses

    def _train_epoch(self):
        # print('_train_epoch')
        dict_metrics_losses = None
        for train_steps_per_epoch, batch in enumerate(self.train_data_loader, 1):
            dict_metrics_losses = self.STRATEGY.run(self._train_step, args=(batch,))
        

        self.epochs += 1

    
        
    def fit(self, train_data_loader, valid_data_loader, saved_path, resume=None):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        while True:
            print('--Stats--')
            print(f'Steps: {self.steps}')
            print(f'Epoch: {self.epochs}')
            self._train_epoch()

            if self.epochs >= self.config['train_max_epochs']:
                print('train_complete')
                break

            self.save_checkpoint()


    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        self._generator = gen_model
        self._discriminator = dis_model
        self._gen_optimizer = gen_optimizer
        self._dis_optimizer = dis_optimizer

        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae_loss = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def save_checkpoint(self):
        """Save checkpoint."""
        # self.ckpt.steps.assign(self.steps)
        # self.ckpt.epochs.assign(self.epochs)
        # self.ckp_manager.save(checkpoint_number=self.steps)
        utils.save_weights(
            self._generator,
            self.saved_path + "generator-{}.h5".format(self.steps)
        )
        utils.save_weights(
            self._discriminator,
            self.saved_path + "discriminator-{}.h5".format(self.steps)
        )