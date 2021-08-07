## 实时语音克隆 - 中文/普通话
![WechatIMG2968](https://user-images.githubusercontent.com/7423248/128490653-f55fefa8-f944-4617-96b8-5cc94f14f8f6.png)

[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat)](http://choosealicense.com/licenses/mit/)
> 该库是从仅支持英语的[Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning) 分叉出来的。

### [English](README.md)  | 中文

## 特性
🌍 **中文** 支持普通话并使用数据集进行测试：adatatang_200zh

🤩 **PyTorch** 适用于 pytorch，已在 1.9.0 版本（最新于 2021 年 8 月）中测试，GPU Tesla T4 和 GTX 2060

🌍 **Windows + Linux** 在修复 nits 后在 Windows 操作系统和 linux 操作系统中进行测试

🤩 **Easy & Awesome** 仅使用新训练的合成器（synthesizer）就有良好效果，复用预训练的编码器/声码器

## 快速开始

### 1. 安装要求
> 按照原始存储库测试您是否已准备好所有环境。
**Python 3.7 或更高版本 ** 需要运行工具箱。

* 安装 [PyTorch](https://pytorch.org/get-started/locally/)。
* 安装 [ffmpeg](https://ffmpeg.org/download.html#get-packages)。
* 运行`pip install -r requirements.txt` 来安装剩余的必要包。


### 2. 使用 aidatatang_200zh 训练合成器
* 下载 adatatang_200zh 数据集并解压：确保您可以访问 *train* 文件夹中的所有 .wav
* 使用音频和梅尔频谱图进行预处理：
`python synthesizer_preprocess_audio.py <datasets_root>`

* 预处理嵌入：
`python synthesizer_preprocess_embeds.py <datasets_root>/SV2TTS/synthesizer`

* 训练合成器：
`python synthesizer_train.py mandarin <datasets_root>/SV2TTS/synthesizer`

* 当您在训练文件夹 *synthesizer/saved_models/* 中看到注意线显示和损失满足您的需要时，请转到下一步。
> 仅供参考，我的注意力是在 18k 步之后出现的，并且在 50k 步之后损失变得低于 0.4。


### 3. 启动工具箱
然后您可以尝试使用工具箱：
`python demo_toolbox.py -d <datasets_root>`

＃＃ TODO
- 添加演示视频
- 添加对更多数据集的支持
- 上传预训练模型
- 🙏 欢迎补充