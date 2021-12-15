# TTS_Text_Mel


## Setup RePo For Learning

1. Download the LJSpeech (dataset)[https://keithito.com/LJ-Speech-Dataset/] or other dataset and format to match LJSpeech. (Move to working directory (src))
2. Complete preprocessing utilizing the TensorflowTTS library using the command below. (Convert characters to IDs and compute mel spectrograms)
```
tensorflow-tts-preprocess --rootdir ./LJSpeech-1.1 --outdir ./dump/dump_ljspeech --config ./conf/ljspeech_preprocess.yaml --dataset ljspeech
```
3. Run the following command to normilize mel spectrograms. ([-1, 1] range)
```
tensorflow-tts-normalize --rootdir ./dump/dump_ljspeech --outdir ./dump/dump_ljspeech --config ./conf/ljspeech_preprocess.yaml --dataset ljspeech 
```
4. Start the training script.
```
python ./ex_train.py --train-dir ./dump/dump_ljspeech/train/ --dev-dir ./dump/dump_ljspeech/valid --outdir ./exp/train.tacotron2.v1/  --config ./conf/tacotron2.v1.yaml --use-norm 1 --mixed_precision 0 --resume ""
```


## Key Files

stats.npy contains the mean and std from the training split mel spectrograms
stats_energy.npy contains the mean and std of energy values from the training split
stats_f0.npy contains the mean and std of F0 values in the training split
train_utt_ids.npy / valid_utt_ids.npy contains training and validation utterances IDs respectively

We use suffix (ids, raw-feats, raw-energy, raw-f0, norm-feats, and wave) for each input type.
