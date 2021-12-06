import sys
import argparse
import logging
import os
import yaml

import tensorflow as tf
import numpy as np

import tensorflow_tts
from tensorflow_tts.inference import TFAutoModel
from tensorflow_tts.configs.tacotron2 import Tacotron2Config
from tensorflow_tts.models import TFTacotron2, TFMelGANMultiScaleDiscriminator

from discriminator import make_discriminator_model
from generator import Generator

from trainer import Trainer
from tacotron_dataset import CharactorMelDataset



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
)

charactorMelDataset = CharactorMelDataset(
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
)

charactorMelDataset.data_graphics()

valid_dataset = charactorMelDataset.create(
    is_shuffle=config["is_shuffle"],
    allow_cache=config["allow_cache"],
    batch_size=config["batch_size"]
)

STRATEGY = tf.distribute.OneDeviceStrategy(device="/gpu:0")

with STRATEGY.scope():
    tacotron_config = Tacotron2Config(**config["tacotron2_params"])
    # tacotron2 = TFTacotron2(config=tacotron_config, name="tacotron2")
    # tacotron2._build()
    # tacotron2.summary()
    tacotron2 = TFAutoModel.from_pretrained("tensorspeech/tts-tacotron2-ljspeech-en", name="tacotron2")
    generator = Generator(config=config, base_model=tacotron2)


    gen_optimizer = tf.keras.optimizers.Adam(**config["generator_optimizer_params"])
    dis_optimizer = tf.keras.optimizers.Adam( **config["discriminator_optimizer_params"])

    discriminator = make_discriminator_model()
    # discriminator.compile(optimizer=dis_optimizer, loss="mse", metrics=["mae"])
    discriminator.summary()

    # define trainer
trainer = Trainer(
    config=config,
    STRATEGY=STRATEGY,
    steps=0,
    epochs=0
)

# compile trainer
trainer.compile(
    gen_model=generator,
    dis_model=discriminator,
    gen_optimizer=gen_optimizer,
    dis_optimizer=dis_optimizer
)

# start training
args.resume = None if len(args.resume) < 1 else args.resume
try:
    trainer.fit(
        train_dataset,
        valid_dataset,
        saved_path=os.path.join(config["outdir"], "checkpoints/"),
        resume_path=args.resume,
    )
except KeyboardInterrupt:
    trainer.save_checkpoint()
    logging.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")
