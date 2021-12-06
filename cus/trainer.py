import os
import json
import wavio

import tensorflow as tf
import numpy as np

from tensorflow_tts.inference import TFAutoModel, AutoProcessor
from tensorflow_tts.utils import utils
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy

class Trainer():

    def __init__(
        self,
        config,
        STRATEGY,
        steps=0,
        epochs=0,
        batches=0
    ):

        self.config = config
        self.STRATEGY = STRATEGY
        self.saved_path = self.config["outdir"] + "/checkpoints/"
        os.makedirs(self.saved_path, exist_ok=True)

        self.steps = steps
        self.epochs = epochs
        self.batches = batches

        self.writer = tf.summary.create_file_writer(config["outdir"])
        
        self.list_metrics_name = [
            "adv_loss",
            "fm_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
        ]

        self.train_metrics = {}
        for name in self.list_metrics_name:
            self.train_metrics.update(
                {name: tf.keras.metrics.Mean(name="train_" + name, dtype=tf.float32)}
            )

        self.melgan = TFAutoModel.from_pretrained("tensorspeech/tts-melgan-ljspeech-en", name="melgan")
        self.tacotron2_processor = AutoProcessor.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en")

    def update_train_metrics(self, dict_metrics_losses):
        for name, value in dict_metrics_losses.items():
            self.train_metrics[name].update_state(value)

    def _check_save_interval(self):
        """Save interval checkpoint."""
        if self.steps % self.config["save_interval_steps"] == 0:
            self.save_checkpoint()
            print(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_log_interval(self, batch):
        """Log to tensorboard."""
        if self.steps % self.config["log_interval_steps"] == 0:
            for metric_name in self.list_metrics_name:
                print(f"(Step: {self.steps}) train_{metric_name} = {self.train_metrics[metric_name].result():.4f}.")
            self._write_to_tensorboard(self.train_metrics, stage="train")
            self.generate_and_save_intermediate_result(batch)

    def _write_to_tensorboard(self, list_metrics, stage="train"):
        """Write variables to tensorboard."""
        with self.writer.as_default():
            for key, value in list_metrics.items():
                tf.summary.scalar(stage + "/" + key, value.result(), step=self.steps)
                self.writer.flush()

    def padd_slice_mel_output(self, mel_outputs):
        # Padd or Slice generator output so that the discrimator can accept the shape
        # print('Padd or Slice')
        # for index in range(len(mel_outputs)):
        #     mel_outputs[index][tf.math.is_nan(mel_outputs[index])] = 0
        mel_outputs_shape = mel_outputs.get_shape()
        add = self.config["max_mel_length"] - mel_outputs_shape[1]
        # Checks if padding/slicing is required
        if add < 0:
            mel_outputs = tf.slice(mel_outputs, [0,0,0], [mel_outputs_shape[0], self.config["max_mel_length"], mel_outputs_shape[2]])
        else:
            paddings = tf.constant([[0, 0], [0, add], [0,0]])
            mel_outputs = tf.pad(mel_outputs, paddings, "CONSTANT")
        
        return mel_outputs

    def _train_step(self, batch):
        print(f'Epoch: {self.epochs}')
        print(f'Steps: {self.steps}')
        print(f'Batches: {self.batches}')
        print(f'UTT_ID: ' + str(batch['utt_ids'][0]))


        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = self._generator(
            **batch,
            training=False
        )
        if tf.math.is_nan(mel_outputs[0][0][0]):
            print(f'##### NONE! #####')
            print(mel_outputs)
            value_not_nan = tf.dtypes.cast(tf.math.logical_not(tf.math.is_nan(mel_outputs)), dtype=tf.float32)
            mel_outputs = tf.math.multiply_no_nan(mel_outputs, value_not_nan)
            print(mel_outputs)



        # print(batch.keys())
        # print(batch['input_ids'][0])
        # print(batch['input_lengths'][0])
        # print(batch['speaker_ids'][0])
        # print(mel_outputs[0])


        mel_outputs = self.padd_slice_mel_output(mel_outputs)

        dict_metrics_losses = {}

        adv_loss, fm_loss = self.train_generator(batch, batch['mel_gts'])
        
        dict_metrics_losses['adv_loss'] = adv_loss
        dict_metrics_losses['fm_loss'] = fm_loss
        dict_metrics_losses['gen_loss'] = adv_loss

        real_loss = self.train_discriminator(batch['mel_gts'], True)
        fake_loss = self.train_discriminator(mel_outputs, False)

        dict_metrics_losses['real_loss'] = real_loss
        dict_metrics_losses['fake_loss'] = fake_loss
        dict_metrics_losses['dis_loss'] = fake_loss + real_loss

        self.update_train_metrics(dict_metrics_losses)

        self.steps += 1

    def train_generator(self, batch, real_data):
        # print(f'train_generator {self.steps}')

        batch_fm_loss = 0.0
        batch_adv_loss = 0.0

        with tf.GradientTape() as gtape:
            
            (
                decoder_output,
                mel_outputs,
                stop_token_predictions,
                alignment_historys,
            ) = self._generator(
                **batch,
                training=False
            )
                
            # print(mel_outputs[0])
            mel_outputs = self.padd_slice_mel_output(mel_outputs)
            
            gtape.watch(mel_outputs)

            p_dis_gen = self._discriminator(mel_outputs)
            p_dis_real = self._discriminator(real_data)
            # print(mel_outputs[0])
            # print(p_dis_gen[0][0])
            # print(p_dis_real[0][0])
            fm_loss = self.mae_loss(p_dis_real, p_dis_gen)
            adv_loss = self.mse_loss(tf.ones_like(p_dis_real), p_dis_gen)
            # print(fm_loss.get_shape())
            # print(adv_loss.get_shape())
            # print(adv_loss)

            adv_loss += self.config["lambda_feat_match"]* fm_loss
            
            # batch_fm_loss += tf.math.reduce_mean(fm_loss).numpy()
            # batch_adv_loss += tf.math.reduce_mean(adv_loss).numpy()

            # print(self._generator.trainable_variables)
            gtape.watch(self._generator.trainable_weights)
            gtape.watch(adv_loss)

            # adv_loss += self.config["lambda_feat_match"] * fm_loss
            # print(self._generator.trainable_variables)
            # print(adv_loss)
            # print(fm_loss)
            gradients = gtape.gradient(
                adv_loss, self._generator.trainable_weights
            )

            self._gen_optimizer.apply_gradients(
                zip(gradients, self._generator.trainable_variables)
            )


        return adv_loss, fm_loss


    def train_discriminator(self, x, real):
        # print(f'train_discriminator {self.steps}')

        batch_loss = 0.0
        with tf.GradientTape() as gtape:

            predictions = self._discriminator(x)
            # print(x.get_shape())
            # print(x[0])
            expected_predictions = tf.zeros_like(predictions) if real else tf.ones_like(predictions)
            loss = self.mse_loss(expected_predictions, predictions)
            # print(expected_predictions.get_shape)
            # print(loss.get_shape())
            # print(loss)

            gtape.watch(self._discriminator.trainable_variables)
            gtape.watch(loss)
            
            gradients = gtape.gradient(
                loss, self._discriminator.trainable_variables
            )
            # print(loss)
            # print('-----------')
            # print(loss.get_shape())
            # print(len(self._discriminator.trainable_weights))
            self._dis_optimizer.apply_gradients(
                zip(gradients, self._discriminator.trainable_variables)
            )


        return loss

    def _train_epoch(self):
        dict_metrics_losses = None
        for train_steps_per_epoch, batch in enumerate(self.train_data_loader, 1):
            dict_metrics_losses = self.STRATEGY.run(self._train_step, args=(batch,))

            self._check_log_interval(batch)
            self._check_save_interval()

            self.batches += 1

            

        self.epochs += 1

    
        
    def fit(self, train_data_loader, valid_data_loader, saved_path, resume_path=None):
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader

        self.create_checkpoint_manager(saved_path=saved_path, max_to_keep=10000)
        if resume_path != None:
            self.load_checkpoint(resume_path)
            logging.info(f"Successfully resumed from {resume_path}.")

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

    def create_checkpoint_manager(self, saved_path=None, max_to_keep=10):
        """Create checkpoint management."""
        if saved_path is None:
            saved_path = self.config["outdir"] + "/checkpoints/"

        os.makedirs(saved_path, exist_ok=True)

        self.saved_path = saved_path
        self.ckpt = tf.train.Checkpoint(
            steps=tf.Variable(1),
            epochs=tf.Variable(1),
            gen_optimizer=self._gen_optimizer,
            dis_optimizer=self._dis_optimizer,
        )
        self.ckp_manager = tf.train.CheckpointManager(
            self.ckpt, saved_path, max_to_keep=max_to_keep
        )

    def save_checkpoint(self):
        """Save checkpoint."""
        self.ckpt.steps.assign(self.steps)
        self.ckpt.epochs.assign(self.epochs)
        self.ckp_manager.save(checkpoint_number=self.steps)
        utils.save_weights(
            self._generator,
            self.saved_path + "generator-{}.h5".format(self.steps)
        )
        utils.save_weights(
            self._discriminator,
            self.saved_path + "discriminator-{}.h5".format(self.steps)
        )

    def load_checkpoint(self, pretrained_path):
        """Load checkpoint."""
        self.ckpt.restore(pretrained_path)
        self.steps = self.ckpt.steps.numpy()
        self.epochs = self.ckpt.epochs.numpy()
        self._gen_optimizer = self.ckpt.gen_optimizer
        # re-assign iterations (global steps) for gen_optimizer.
        self._gen_optimizer.iterations.assign(tf.cast(self.steps, tf.int64))
        # re-assign iterations (global steps) for dis_optimizer.
        try:
            discriminator_train_start_steps = self.config["discriminator_train_start_steps"]
            discriminator_train_start_steps = tf.math.maximum(
                0, self.steps - discriminator_train_start_steps 
            )
        except Exception:
            discriminator_train_start_steps = self.steps
        self._dis_optimizer = self.ckpt.dis_optimizer
        self._dis_optimizer.iterations.assign(
            tf.cast(discriminator_train_start_steps, tf.int64)
        )

        # load weights.
        utils.load_weights(
            self._generator,
            self.saved_path + "generator-{}.h5".format(self.steps)
        )
        utils.load_weights(
            self._discriminator,
            self.saved_path + "discriminator-{}.h5".format(self.steps)
        )


    def generate_and_save_intermediate_result(self, batch):
        print('generate_and_save_intermediate_result')
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function for faster.
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = self._generator(**batch, training=False)
        mel_gts = batch["mel_gts"]
        utt_ids = batch["utt_ids"]

        # convert to tensor.
        # here we just take a sample at first replica.
        try:
            mels_before = decoder_output.values[0].numpy()
            mels_after = mel_outputs.values[0].numpy()
            mel_gts = mel_gts.values[0].numpy()
            alignment_historys = alignment_historys.values[0].numpy()
            utt_ids = utt_ids.values[0].numpy()
        except Exception:
            mels_before = decoder_output.numpy()
            mels_after = mel_outputs.numpy()
            mel_gts = mel_gts.numpy()
            alignment_historys = alignment_historys.numpy()
            utt_ids = utt_ids.numpy()

        # check directory
        dirname = os.path.join(self.config["outdir"], f"predictions/{self.steps}steps")
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        print(dirname)

        for idx, (mel_gt, mel_before, mel_after, alignment_history) in enumerate(zip(mel_gts, mels_before, mels_after, alignment_historys), 0):
            mel_gt = tf.reshape(mel_gt, (-1, 80)).numpy()  # [length, 80]
            mel_before = tf.reshape(mel_before, (-1, 80)).numpy()  # [length, 80]
            mel_after = tf.reshape(mel_after, (-1, 80)).numpy()  # [length, 80]

            # plot figure and save it
            utt_id = utt_ids[idx]
            figname = os.path.join(dirname, f"{utt_id}.png")
            fig = plt.figure(figsize=(10, 8))
            ax1 = fig.add_subplot(311)
            ax2 = fig.add_subplot(312)
            ax3 = fig.add_subplot(313)
            im = ax1.imshow(np.rot90(mel_gt), aspect="auto", interpolation="none")
            ax1.set_title("Target Mel-Spectrogram")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax1)
            ax2.set_title(f"Predicted Mel-before-Spectrogram @ {self.steps} steps")
            im = ax2.imshow(np.rot90(mel_before), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax2)
            ax3.set_title(f"Predicted Mel-after-Spectrogram @ {self.steps} steps")
            im = ax3.imshow(np.rot90(mel_after), aspect="auto", interpolation="none")
            fig.colorbar(mappable=im, shrink=0.65, orientation="horizontal", ax=ax3)
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()

            # plot alignment
            figname = os.path.join(dirname, f"{idx}_alignment.png")
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111)
            ax.set_title(f"Alignment @ {self.steps} steps")
            im = ax.imshow(
                alignment_history, aspect="auto", origin="lower", interpolation="none"
            )
            fig.colorbar(im, ax=ax)
            xlabel = "Decoder timestep"
            plt.xlabel(xlabel)
            plt.ylabel("Encoder timestep")
            plt.tight_layout()
            plt.savefig(figname)
            plt.close()


        # Generate Waveform
        # print(mel_outputs[idx].get_shape())
        # filename = os.path.join(dirname, f"training_data_audio_sample.wav")
        # audio = self.melgan(mel_outputs)[0, :, 0].numpy()
        # wavio.write(filename, audio, 22050, sampwidth=3)

        input_ids = self.tacotron2_processor.text_to_sequence('Hello my name is Joe. Nice to meet you!')

        _, mel_outputs, stop_token_prediction, alignment_history = self._generator.inference(
            tf.expand_dims(tf.convert_to_tensor(input_ids, dtype=tf.int32), 0),
            tf.convert_to_tensor([len(input_ids)], tf.int32),
            tf.convert_to_tensor([0], dtype=tf.int32)
        )
        print('mel_outputs')
        print(mel_outputs)

        filename = os.path.join(dirname, f"audio_sample.wav")
        audio = self.melgan(mel_outputs)[0, :, 0].numpy()
        wavio.write(filename, audio, 22050, sampwidth=3)