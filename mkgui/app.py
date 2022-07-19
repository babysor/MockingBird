from pydantic import BaseModel, Field
import os
from pathlib import Path
from enum import Enum
from encoder import inference as encoder
import librosa
from scipy.io.wavfile import write
import re
import numpy as np
from mkgui.base.components.types import FileContent
from vocoder.hifigan import inference as gan_vocoder
from synthesizer.inference import Synthesizer
from typing import Any, Tuple
import matplotlib.pyplot as plt

# Constants
AUDIO_SAMPLES_DIR = f"samples{os.sep}"
SYN_MODELS_DIRT = f"synthesizer{os.sep}saved_models"
ENC_MODELS_DIRT = f"encoder{os.sep}saved_models"
VOC_MODELS_DIRT = f"vocoder{os.sep}saved_models"
TEMP_SOURCE_AUDIO = f"wavs{os.sep}temp_source.wav"
TEMP_RESULT_AUDIO = f"wavs{os.sep}temp_result.wav"
if not os.path.isdir("wavs"):
    os.makedirs("wavs")

# Load local sample audio as options TODO: load dataset 
if os.path.isdir(AUDIO_SAMPLES_DIR):
    audio_input_selection = Enum('samples', list((file.name, file) for file in Path(AUDIO_SAMPLES_DIR).glob("*.wav")))
# Pre-Load models
if os.path.isdir(SYN_MODELS_DIRT):    
    synthesizers =  Enum('synthesizers', list((file.name, file) for file in Path(SYN_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded synthesizer models: " + str(len(synthesizers)))
else:
    raise Exception(f"Model folder {SYN_MODELS_DIRT} doesn't exist.")

if os.path.isdir(ENC_MODELS_DIRT):    
    encoders =  Enum('encoders', list((file.name, file) for file in Path(ENC_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded encoders models: " + str(len(encoders)))
else:
    raise Exception(f"Model folder {ENC_MODELS_DIRT} doesn't exist.")

if os.path.isdir(VOC_MODELS_DIRT):    
    vocoders =  Enum('vocoders', list((file.name, file) for file in Path(VOC_MODELS_DIRT).glob("**/*gan*.pt")))
    print("Loaded vocoders models: " + str(len(synthesizers)))
else:
    raise Exception(f"Model folder {VOC_MODELS_DIRT} doesn't exist.")



class Input(BaseModel):
    message: str = Field(
        ..., example="欢迎使用工具箱, 现已支持中文输入！", alias="文本内容"
    )
    local_audio_file: audio_input_selection = Field(
        ..., alias="输入语音（本地wav）",
        description="选择本地语音文件."
    )
    upload_audio_file: FileContent = Field(default=None, alias="或上传语音",
        description="拖拽或点击上传.", mime_type="audio/wav")
    encoder: encoders = Field(
        ..., alias="编码模型", 
        description="选择语音编码模型文件."
    )
    synthesizer: synthesizers = Field(
        ..., alias="合成模型", 
        description="选择语音合成模型文件."
    )
    vocoder: vocoders = Field(
        ..., alias="语音解码模型", 
        description="选择语音解码模型文件(目前只支持HifiGan类型)."
    )

class AudioEntity(BaseModel):
    content: bytes
    mel: Any

class Output(BaseModel):
    __root__: Tuple[AudioEntity, AudioEntity]

    def render_output_ui(self, streamlit_app, input) -> None:  # type: ignore
        """Custom output UI.
        If this method is implmeneted, it will be used instead of the default Output UI renderer.
        """
        src, result = self.__root__
        
        streamlit_app.subheader("Synthesized Audio")
        streamlit_app.audio(result.content, format="audio/wav")

        fig, ax = plt.subplots()
        ax.imshow(src.mel, aspect="equal", interpolation="none")
        ax.set_title("mel spectrogram(Source Audio)")
        streamlit_app.pyplot(fig)
        fig, ax = plt.subplots()
        ax.imshow(result.mel, aspect="equal", interpolation="none")
        ax.set_title("mel spectrogram(Result Audio)")
        streamlit_app.pyplot(fig)


def synthesize(input: Input) -> Output:
    """synthesize(合成)"""
    # load models
    encoder.load_model(Path(input.encoder.value))
    current_synt = Synthesizer(Path(input.synthesizer.value))
    gan_vocoder.load_model(Path(input.vocoder.value))

    # load file
    if input.upload_audio_file != None:
        with open(TEMP_SOURCE_AUDIO, "w+b") as f:
            f.write(input.upload_audio_file.as_bytes())
            f.seek(0)
        wav, sample_rate = librosa.load(TEMP_SOURCE_AUDIO)
    else:
        wav, sample_rate  = librosa.load(input.local_audio_file.value)
        write(TEMP_SOURCE_AUDIO, sample_rate, wav) #Make sure we get the correct wav

    source_spec = Synthesizer.make_spectrogram(wav)

    # preprocess
    encoder_wav = encoder.preprocess_wav(wav, sample_rate)
    embed, _, _ = encoder.embed_utterance(encoder_wav, return_partials=True)

    # Load input text
    texts = filter(None, input.message.split("\n"))
    punctuation = '！，。、,' # punctuate and split/clean text
    processed_texts = []
    for text in texts:
        for processed_text in re.sub(r'[{}]+'.format(punctuation), '\n', text).split('\n'):
            if processed_text:
                processed_texts.append(processed_text.strip())
    texts = processed_texts

    # synthesize and vocode
    embeds = [embed] * len(texts)
    specs = current_synt.synthesize_spectrograms(texts, embeds)
    spec = np.concatenate(specs, axis=1)
    sample_rate = Synthesizer.sample_rate
    wav, sample_rate = gan_vocoder.infer_waveform(spec)

    # write and output 
    write(TEMP_RESULT_AUDIO, sample_rate, wav) #Make sure we get the correct wav
    with open(TEMP_SOURCE_AUDIO, "rb") as f:
        source_file = f.read()
    with open(TEMP_RESULT_AUDIO, "rb") as f:
        result_file = f.read()
    return Output(__root__=(AudioEntity(content=source_file, mel=source_spec), AudioEntity(content=result_file, mel=spec)))