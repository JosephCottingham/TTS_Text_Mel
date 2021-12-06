import argparse



"""Run."""
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
    "--output-dir",
    default=None,
    type=str,
    help="directory where all graphics will be output. ",
)
parser.add_argument(
    "--config", type=str, required=True, help="yaml format configuration file."
)

args = parser.parse_args()

if args.train_dir is None:
    raise ValueError("Please specify --train-dir")
if args.output_dir is None:
    raise ValueError("Please specify --output-dir")
if args.config is None:
    raise ValueError("Please specify --config")


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