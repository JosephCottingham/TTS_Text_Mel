# -*- coding: utf-8 -*-
"""Train Tacotron2Gan."""
import tensorflow as tf

physical_devices = tf.config.list_physical_devices("GPU")
for i in range(len(physical_devices)):
    tf.config.experimental.set_memory_growth(physical_devices[i], True)

import sys

sys.path.append(".")

import argparse
import logging
import os

import numpy as np
import yaml
from tqdm import tqdm

import tensorflow_tts
from tacotron_dataset import CharactorMelDataset
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
import tensorflow_tts.configs.melgan as MELGAN_CONFIG
from tensorflow_tts.models import TFTacotron2, TFMelGANMultiScaleDiscriminator
from tensorflow_tts.optimizers import AdamWeightDecay, WarmUp
from tensorflow_tts.trainers import GanBasedTrainer
from tensorflow_tts.utils import calculate_2d_loss, calculate_3d_loss, return_strategy
from Discriminator import make_discriminator_model

class Tacotron2Trainer(GanBasedTrainer):
    """Tacotron2 Trainer class based on GanBasedTrainer."""

    def __init__(
        self,
        config,
        strategy,
        steps=0,
        epochs=0,
        is_generator_mixed_precision=False,
        is_discriminator_mixed_precision=False
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            config (dict): Config dict loaded from yaml format configuration file.
            is_generator_mixed_precision (bool): Use mixed precision for generator or not.
            is_discriminator_mixed_precision (bool): Use mixed precision for discriminator or not.
        """
        super(Tacotron2Trainer, self).__init__(
            steps=steps,
            epochs=epochs,
            config=config,
            strategy=strategy,
            is_generator_mixed_precision=is_generator_mixed_precision,
            is_discriminator_mixed_precision=is_discriminator_mixed_precision,
        )

        # define metrics to aggregates data and use tf.summary logs them
        self.list_metrics_name = [
            "adversarial_loss",
            "fm_loss",
            "gen_loss",
            "real_loss",
            "fake_loss",
            "dis_loss",
            "mels_spectrogram_loss",
        ]

        # old Metrics used for tacotron2
        # define metrics to aggregates data and use tf.summary logs them
        # self.list_metrics_name = [
        #     "stop_token_loss",
        #     "mel_loss_before",
        #     "mel_loss_after",
        #     "guided_attention_loss",
        # ]

        self.init_train_eval_metrics(self.list_metrics_name)
        self.reset_states_train()
        self.reset_states_eval()

        self.config = config

    def compile(self, gen_model, dis_model, gen_optimizer, dis_optimizer):
        super().compile(gen_model, dis_model, gen_optimizer, dis_optimizer)
        self.binary_crossentropy = tf.keras.losses.BinaryCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE
        )
        self.mse_loss = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.NONE
        )
        self.mae_loss = tf.keras.losses.MeanAbsoluteError(
            reduction=tf.keras.losses.Reduction.NONE
        )

    def compute_per_example_generator_losses(self, batch, outputs):
        """Compute per example generator losses and return dict_metrics_losses
        Note that all element of the loss MUST has a shape [batch_size] and 
        the keys of dict_metrics_losses MUST be in self.list_metrics_name.
        Args:
            batch: dictionary batch input return from dataloader
            outputs: outputs of the model
        
        Returns:
            per_example_losses: per example losses for each GPU, shape [B]
            dict_metrics_losses: dictionary loss.
        """
        mel_gts = batch["mel_gts"]
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs

        # [[32, None, 80], [32, None, 80], [32, None], [32, 188, None]]


        p_hat = self._discriminator(mel_outputs)

        p = tf.squeeze(self._discriminator(tf.expand_dims(mel_gts, 2)))
        adv_loss = 0.0
        for i in range(len(p_hat)):
            print(p_hat[i])


        for i in range(len(p_hat)):
            adv_loss += calculate_2d_loss(
                tf.ones_like(p_hat[i]), p_hat[i], loss_fn=self.mse_loss
            )
        adv_loss /= i + 1

        # define feature-matching loss
        fm_loss = 0.0
        for i in range(len(p_hat)):
            fm_loss += calculate_2d_loss(
                p[i], p_hat[i], loss_fn=self.mae_loss
            )
        fm_loss /= (i + 1)
        adv_loss += self.config["lambda_feat_match"] * fm_loss

        per_example_losses = adv_loss

        dict_metrics_losses = {
            "adversarial_loss": adv_loss,
            "fm_loss": fm_loss,
            "gen_loss": adv_loss
        }

        return per_example_losses, dict_metrics_losses


    def compute_per_example_discriminator_losses(self, batch, gen_outputs):
        mel_gts = batch["mel_gts"]
        y_hat = gen_outputs

        y = tf.expand_dims(mel_gts, 2)
        p = self._discriminator(y)
        p_hat = self._discriminator(y_hat)

        real_loss = 0.0
        fake_loss = 0.0
        for i in range(len(p)):
            real_loss += calculate_3d_loss(
                tf.ones_like(p[i][-1]), p[i][-1], loss_fn=self.mse_loss
            )
            fake_loss += calculate_3d_loss(
                tf.zeros_like(p_hat[i][-1]), p_hat[i][-1], loss_fn=self.mse_loss
            )
        real_loss /= i + 1
        fake_loss /= i + 1
        dis_loss = real_loss + fake_loss

        # calculate per_example_losses and dict_metrics_losses
        per_example_losses = dis_loss

        dict_metrics_losses = {
            "real_loss": real_loss,
            "fake_loss": fake_loss,
            "dis_loss": dis_loss,
        }

        return per_example_losses, dict_metrics_losses

    def generate_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        import matplotlib.pyplot as plt

        # predict with tf.function for faster.
        outputs = self.one_step_predict(batch)
        (
            decoder_output,
            mel_outputs,
            stop_token_predictions,
            alignment_historys,
        ) = outputs
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

        for idx, (mel_gt, mel_before, mel_after, alignment_history) in enumerate(
            zip(mel_gts, mels_before, mels_after, alignment_historys), 0
        ):
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


def main():
    """Run training process."""
    parser = argparse.ArgumentParser(
        description="Train FastSpeech (See detail in tensorflow_tts/bin/train-fastspeech.py)"
    )
    parser.add_argument(
        "--train-dir",
        default=None,
        type=str,
        help="directory including training data. ",
    )
    parser.add_argument(
        "--dev-dir",
        default=None,
        type=str,
        help="directory including development data. ",
    )
    parser.add_argument(
        "--use-norm", default=1, type=int, help="usr norm-mels for train or raw."
    )
    parser.add_argument(
        "--outdir", type=str, required=True, help="directory to save checkpoints."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="yaml format configuration file."
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        nargs="?",
        help='checkpoint file path to resume training. (default="")',
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="logging level. higher is more logging. (default=1)",
    )
    parser.add_argument(
        "--mixed_precision",
        default=0,
        type=int,
        help="using mixed precision for generator or not.",
    )
    parser.add_argument(
        "--pretrained",
        default="",
        type=str,
        nargs="?",
        help="pretrained weights .h5 file to load weights from. Auto-skips non-matching layers",
    )
    parser.add_argument(
        "--use-fal",
        default=0,
        type=int,
        help="Use forced alignment guided attention loss or regular",
    )
    args = parser.parse_args()

    # return strategy
    STRATEGY = return_strategy()

    # set mixed precision config
    if args.mixed_precision == 1:
        tf.config.optimizer.set_experimental_options({"auto_mixed_precision": True})

    args.mixed_precision = bool(args.mixed_precision)
    args.use_norm = bool(args.use_norm)
    args.use_fal = bool(args.use_fal)

    # set logger
    if args.verbose > 1:
        logging.basicConfig(
            level=logging.DEBUG,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose > 0:
        logging.basicConfig(
            level=logging.INFO,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            stream=sys.stdout,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check directory existence
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    # check arguments
    if args.train_dir is None:
        raise ValueError("Please specify --train-dir")
    if args.dev_dir is None:
        raise ValueError("Please specify --valid-dir")

    # load and save config
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.Loader)
    config.update(vars(args))
    config["version"] = tensorflow_tts.__version__

    # get dataset
    if config["remove_short_samples"]:
        mel_length_threshold = config["mel_length_threshold"]
    else:
        mel_length_threshold = 0

    if config["format"] == "npy":
        charactor_query = "*-ids.npy"
        mel_query = "*-raw-feats.npy" if args.use_norm is False else "*-norm-feats.npy"
        align_query = "*-alignment.npy" if args.use_fal is True else ""
        charactor_load_fn = np.load
        mel_load_fn = np.load
    else:
        raise ValueError("Only npy are supported.")

    train_dataset = CharactorMelDataset(
        dataset=config["tacotron2_params"]["dataset"],
        root_dir=args.train_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        use_fixed_shapes=config["use_fixed_shapes"],
        align_query=align_query,
    )

    # update max_mel_length and max_char_length to config
    config.update({"max_mel_length": int(train_dataset.max_mel_length)})
    config.update({"max_char_length": int(train_dataset.max_char_length)})

    with open(os.path.join(args.outdir, "config.yml"), "w") as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    for key, value in config.items():
        logging.info(f"{key} = {value}")

    train_dataset = train_dataset.create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"]
        * STRATEGY.num_replicas_in_sync
        * config["gradient_accumulation_steps"],
    )

    valid_dataset = CharactorMelDataset(
        dataset=config["tacotron2_params"]["dataset"],
        root_dir=args.dev_dir,
        charactor_query=charactor_query,
        mel_query=mel_query,
        charactor_load_fn=charactor_load_fn,
        mel_load_fn=mel_load_fn,
        mel_length_threshold=mel_length_threshold,
        reduction_factor=config["tacotron2_params"]["reduction_factor"],
        use_fixed_shapes=False,  # don't need apply fixed shape for evaluation.
        align_query=align_query,
    ).create(
        is_shuffle=config["is_shuffle"],
        allow_cache=config["allow_cache"],
        batch_size=config["batch_size"] * STRATEGY.num_replicas_in_sync,
    )

    # define trainer
    trainer = Tacotron2Trainer(
        config=config,
        strategy=STRATEGY,
        steps=0,
        epochs=0,
        is_generator_mixed_precision=args.mixed_precision,
        is_discriminator_mixed_precision=args.mixed_precision
    )

    with STRATEGY.scope():
        # define model.
        tacotron_config = Tacotron2Config(**config["tacotron2_params"])
        tacotron2 = TFTacotron2(config=tacotron_config, name="tacotron2")
        tacotron2._build()
        tacotron2.summary()

    #     if len(args.pretrained) > 1:
    #         tacotron2.load_weights(args.pretrained, by_name=True, skip_mismatch=True)
    #         logging.info(
    #             f"Successfully loaded pretrained weight from {args.pretrained}."
    #         )

    #     # AdamW for tacotron2
    #     learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
    #         initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
    #         decay_steps=config["optimizer_params"]["decay_steps"],
    #         end_learning_rate=config["optimizer_params"]["end_learning_rate"],
    #     )

    #     learning_rate_fn = WarmUp(
    #         initial_learning_rate=config["optimizer_params"]["initial_learning_rate"],
    #         decay_schedule_fn=learning_rate_fn,
    #         warmup_steps=int(
    #             config["train_max_steps"]*config["optimizer_params"]["warmup_proportion"]
    #         ),
    #     )

    #     optimizer = AdamWeightDecay(
    #         learning_rate=learning_rate_fn,
    #         weight_decay_rate=config["optimizer_params"]["weight_decay"],
    #         beta_1=0.9,
    #         beta_2=0.98,
    #         epsilon=1e-6,
    #         exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"],
    #     )

    #     _ = optimizer.iterations

    gen_optimizer = tf.keras.optimizers.Adam(**config["generator_optimizer_params"])
    dis_optimizer = tf.keras.optimizers.Adam( **config["discriminator_optimizer_params"])

    discriminator = make_discriminator_model()
    discriminator.summary()

    # compile trainer
    trainer.compile(gen_model=tacotron2, dis_model=discriminator, gen_optimizer=gen_optimizer, dis_optimizer=dis_optimizer)

    # start training
    try:
        trainer.fit(
            train_dataset,
            valid_dataset,
            saved_path=os.path.join(config["outdir"], "checkpoints/"),
            resume=args.resume,
        )
    except KeyboardInterrupt:
        trainer.save_checkpoint()
        logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


if __name__ == "__main__":
    main()
