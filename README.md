Step 1: Clone the repository

```
git clone https://github.com/newgenai79/talking_face_preprocessing_windows
```

Step 2: Navigate inside the cloned repository

```
cd talking_face_preprocessing_windows
```

Step 3: Create virtual environment

```
conda create -n tfpw python==3.9.0
```

Step 4: Activate virtual environment

```
conda activate tfpw
```

Step 5: Install requirements

```
pip install -r requirements.txt
```

Step 6: Download weights

```
1. Open new command prompt
2. md weights
3. cd weights
4. git lfs install
5. git clone https://huggingface.co/TencentGameMate/chinese-hubert-large
```

Step 7: Inference

```
python extract_audio_features.py --model_path "weights/chinese-hubert-large" --audio_dir_path "./data_processing/specified_formats/audios/audios_16k/" --audio_feature_saved_path "./data_processing/specified_formats/audios/hubert_features/" --computed_device "cuda" --padding_to_align_audio True
```

## Audio Feature Extraction

### MFCC  (100 hz)

MFCC stands for Mel-frequency cepstral coefficients. It can quickly help us with code testing without the need to install many environments. The output shape of audio_feature will be `(T, 39)`. This feature is not robust and is only suitable for early code testing. For detailed usage, please refer to [mfcc_feature_example.py](libs/mfcc_feature_example.py).


### Hubert Feature with Weighted-sum  (50 hz)

Before extraction, please make sure that all audio files have a sampling rate of `16k` Hz. and download the weights from [URL](https://github.com/TencentGameMate/chinese_speech_pretrain) and put them into weights dir. Although this model was pre-trained on 10,000 hours of Chinese data as unsupervised training data, we have also found that it can generalize to other languages as well.


```bash 

python extract_audio_features.py \
  --model_path "weights/chinese-hubert-large" \
  --audio_dir_path "./data_processing/specified_formats/audios/audios_16k/" \
  --audio_feature_saved_path "./data_processing/specified_formats/audios/hubert_features/" \
  --computed_device "cuda" \
  --padding_to_align_audio True

```

* The purpose of padding_to_align_audio is to pad the end of the audio to match the dimensionality, with the goal of maintaining consistency with video frames for convenient training.
* The result shape is `(25, T, 1024)`, 25 means all hidden layers including the one audio feature extraction plus 24 hidden layers. You can change code get specific layers, such as last layer, for training.
* The purpose for extract all layers is that we trained on `weighted sum` strategies in [diffdub](https://github.com/liutaocode/DiffDub) and [anitalker](https://github.com/X-LANCE/AniTalker).
* Currently, we only have tested feature extraction on hubert model.
* If your audio is long ( > 120 seconds), please set computed_device from `cuda` to `cpu` to avoid GPU out-of-memory.

## Notes

The examples provided herein are based on the HDTF or VoxCeleb datasets and are intended solely for educational and academic research purposes. Please do not use them for any other purposes.

## Acknowledgements

* https://github.com/MRzzm/HDTF
* https://github.com/TencentGameMate/chinese_speech_pretrain
* https://github.com/DefTruth/torchlm
* https://github.com/TadasBaltrusaitis/OpenFace
* https://github.com/cleardusk/3DDFA_V2
* https://www.robots.ox.ac.uk/~vgg/data/voxceleb/
* https://github.com/cpdu/unicats
* https://github.com/X-LANCE/SLAM-LLM