![mockingbird](https://user-images.githubusercontent.com/12797292/131216767-6eb251d6-14fc-4951-8324-2722f0cd4c63.jpg)


[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)

> English | [中文](README-CN.md)

## Features
🌍 **Chinese** supported mandarin and tested with multiple datasets: aidatatang_200zh, magicdata, aishell3

🤩 **PyTorch** worked for pytorch, tested in version of 1.9.0(latest in August 2021), with GPU Tesla T4 and GTX 2060

🌍 **Windows + Linux** tested in both Windows OS and linux OS after fixing nits

🤩 **Easy & Awesome** effect with only newly-trained synthesizer, by reusing the pretrained encoder/vocoder


### [DEMO VIDEO](https://www.bilibili.com/video/BV1sA411P7wM/)

## Quick Start

### 1. Install Requirements
> Follow the original repo to test if you got all environment ready.
**Python 3.7 or higher ** is needed to run the toolbox.

* Install [PyTorch](https://pytorch.org/get-started/locally/).
> If you get an `ERROR: Could not find a version that satisfies the requirement torch==1.9.0+cu102 (from versions: 0.1.2, 0.1.2.post1, 0.1.2.post2 )` This error is probably due to a low version of python, try using 3.9 and it will install successfully
* Install [ffmpeg](https://ffmpeg.org/download.html#get-packages).
* Run `pip install -r requirements.txt` to install the remaining necessary packages.
* Install webrtcvad `pip install webrtcvad-wheels`(If you need)
> Note that we are using the pretrained encoder/vocoder but synthesizer, since the original model is incompatible with the Chinese sympols. It means the demo_cli is not working at this moment.
### 2. Train synthesizer with your dataset
* Download aidatatang_200zh or other dataset and unzip: make sure you can access all .wav in *train* folder
* Preprocess with the audios and the mel spectrograms:
`python pre.py <datasets_root>`
Allow parameter `--dataset {dataset}` to support adatatang_200zh, magicdata, aishell3

>If it happens `the page file is too small to complete the operation`, please refer to this [video](https://www.youtube.com/watch?v=Oh6dga-Oy10&ab_channel=CodeProf) and change the virtual memory to 100G (102400), for example : When the file is placed in the D disk, the virtual memory of the D disk is changed.


* Train the synthesizer:
`python synthesizer_train.py mandarin <datasets_root>/SV2TTS/synthesizer`

* Go to next step when you see attention line show and loss meet your need in training folder *synthesizer/saved_models/*.
> FYI, my attention came after 18k steps and loss became lower than 0.4 after 50k steps.
![attention_step_20500_sample_1](https://user-images.githubusercontent.com/7423248/128587252-f669f05a-f411-4811-8784-222156ea5e9d.png)
![step-135500-mel-spectrogram_sample_1](https://user-images.githubusercontent.com/7423248/128587255-4945faa0-5517-46ea-b173-928eff999330.png)

### 2.2 Use pretrained model of synthesizer
> Thanks to the community, some models will be shared:

| author | Download link | Previow Video |
| --- | ----------- | ----- |
|@miven| https://pan.baidu.com/s/1PI-hM3sn5wbeChRryX-RCQ code：2021 | https://www.bilibili.com/video/BV1uh411B7AD/

> A link to my early trained model: [Baidu Yun](https://pan.baidu.com/s/10t3XycWiNIg5dN5E_bMORQ)
Code：aid4

### 2.3 Train vocoder (Optional)
* Preprocess the data:
`python vocoder_preprocess.py <datasets_root>`

* Train the vocoder:
`python vocoder_train.py mandarin <datasets_root>`

### 3. Launch the Toolbox
You can then try the toolbox:

`python demo_toolbox.py -d <datasets_root>`
or
`python demo_toolbox.py`

> Good news🤩: Chinese Characters are supported

## TODO
- [x] Add demo video
- [X] Add support for more dataset
- [X] Upload pretrained model
- [ ] Support parallel tacotron
- [ ] Service orianted and docterize
- 🙏 Welcome to add more

## Reference
> This repository is forked from [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) which only support English.

| URL | Designation | Title | Implementation source |
| --- | ----------- | ----- | --------------------- |
|[**1806.04558**](https://arxiv.org/pdf/1806.04558.pdf) | **SV2TTS** | **Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis** | This repo |
|[1802.08435](https://arxiv.org/pdf/1802.08435.pdf) | WaveRNN (vocoder) | Efficient Neural Audio Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN) |
|[1703.10135](https://arxiv.org/pdf/1703.10135.pdf) | Tacotron (synthesizer) | Tacotron: Towards End-to-End Speech Synthesis | [fatchord/WaveRNN](https://github.com/fatchord/WaveRNN)
|[1710.10467](https://arxiv.org/pdf/1710.10467.pdf) | GE2E (encoder)| Generalized End-To-End Loss for Speaker Verification | This repo |