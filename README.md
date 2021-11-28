# TTS_Text_Mel



Preprocess audio features
- Convert characters to IDs
- Compute mel spectrograms
- Normalize mel spectrograms to [-1, 1] range
- Split the dataset into train and validation
- Compute the mean and standard deviation of multiple features from the training split

Standardize mel spectrogram based on computed statistics


```
tensorflow-tts-preprocess --rootdir ./[ljspeech/kss/baker/libritts/thorsten/synpaflex] --outdir ./dump_[ljspeech/kss/baker/libritts/thorsten/synpaflex] --config preprocess/[ljspeech/kss/baker/thorsten/synpaflex]_preprocess.yaml --dataset [ljspeech/kss/baker/libritts/thorsten/synpaflex]
tensorflow-tts-normalize --rootdir ./dump_[ljspeech/kss/baker/libritts/thorsten/synpaflex] --outdir ./dump_[ljspeech/kss/baker/libritts/thorsten/synpaflex] --config preprocess/[ljspeech/kss/baker/libritts/thorsten/synpaflex]_preprocess.yaml --dataset [ljspeech/kss/baker/libritts/thorsten/synpaflex]
```


stats.npy contains the mean and std from the training split mel spectrograms
stats_energy.npy contains the mean and std of energy values from the training split
stats_f0.npy contains the mean and std of F0 values in the training split
train_utt_ids.npy / valid_utt_ids.npy contains training and validation utterances IDs respectively

We use suffix (ids, raw-feats, raw-energy, raw-f0, norm-feats, and wave) for each input type.

## Example:
```
tensorflow-tts-preprocess --rootdir ./LJSpeech-1.1 --outdir ./dump/dump_ljspeech --config ./conf/ljspeech_preprocess.yaml --dataset ljspeech

tensorflow-tts-normalize --rootdir ./dump/dump_ljspeech --outdir ./dump/dump_ljspeech --config ./conf/ljspeech_preprocess.yaml --dataset ljspeech 
```

```
tensorflow-tts-preprocess --rootdir C:\Users\josep\Projects\TTS_Text_Mel\training_data\LJSpeech-1.1 --outdir ./dump/dump_ljspeech --config ./configs/jspeech_preprocess.yaml --dataset ljspeech 
tensorflow-tts-normalize --rootdir ./dump/dump_ljspeech --outdir ./dump/dump_ljspeech --config ./configs/jspeech_preprocess.yaml --dataset ljspeech 
```

python ./ex_train.py --train-dir ./dump/dump_ljspeech/train/ --dev-dir ./dump/dump_ljspeech/valid --outdir ./exp/train.tacotron2.v1/  --config ./conf/tacotron2.v1.yaml --use-norm 1 --mixed_precision 0 --resume ""

SET PATH=C:\tools\cuda\bin;%PATH%
SET PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10\bin;%PATH%
