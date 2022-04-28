from asyncio.windows_events import NULL
from pydantic import BaseModel, Field
import os
from pathlib import Path
from enum import Enum
from encoder import inference as encoder
import librosa
from scipy.io.wavfile import write
import re
import numpy as np
from opyrator.components.types import FileContent
from vocoder.hifigan import inference as gan_vocoder
from synthesizer.inference import Synthesizer

# Constants
AUDIO_SAMPLES_DIR = 'samples\\'
SYN_MODELS_DIRT = "synthesizer\\saved_models"
ENC_MODELS_DIRT = "encoder\\saved_models"
VOC_MODELS_DIRT = "vocoder\\saved_models"
TEMP_SOURCE_AUDIO = "wavs/temp_source.wav"
TEMP_RESULT_AUDIO = "wavs/temp_result.wav"

# Load local sample audio as options TODO: load dataset 
if os.path.isdir(AUDIO_SAMPLES_DIR):
    audio_input_selection = Enum('samples', list((file.name, file) for file in Path(AUDIO_SAMPLES_DIR).glob("*.wav")))
# Pre-Load models
if os.path.isdir(SYN_MODELS_DIRT):    
    synthesizers =  Enum('synthesizers', list((file.name, file) for file in Path(SYN_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded synthesizer models: " + str(len(synthesizers)))
if os.path.isdir(ENC_MODELS_DIRT):    
    encoders =  Enum('encoders', list((file.name, file) for file in Path(ENC_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded encoders models: " + str(len(encoders)))
if os.path.isdir(VOC_MODELS_DIRT):    
    vocoders =  Enum('vocoders', list((file.name, file) for file in Path(VOC_MODELS_DIRT).glob("**/*gan*.pt")))
    print("Loaded vocoders models: " + str(len(synthesizers)))


class Input(BaseModel):
    local_audio_file: audio_input_selection = Field(
        ..., alias="输入语音（本地wav）",
        description="选择本地语音文件."
    )
    upload_audio_file: FileContent = Field(..., alias="或上传语音",
        description="拖拽或点击上传.", mime_type="audio/wav")
    encoder: encoders = Field(
        ..., alias="编码模型", 
        description="选择语音编码模型文件."
    )
    synthesizer: synthesizers = Field(
        ..., alias="合成模型", 
        description="选择语音编码模型文件."
    )
    vocoder: vocoders = Field(
        ..., alias="语音编码模型", 
        description="选择语音编码模型文件(目前只支持HifiGan类型)."
    )
    message: str = Field(
        ..., example="欢迎使用工具箱, 现已支持中文输入！", alias="输出文本内容"
    )

class Output(BaseModel):
    result_file: FileContent = Field(
        ...,
        mime_type="audio/wav",
        description="输出音频",
    )
    source_file: FileContent = Field(
        ...,
        mime_type="audio/wav",
        description="原始音频.",
    )

def mocking_bird(input: Input) -> Output:
    """欢迎使用MockingBird Web 2"""
    # load models
    encoder.load_model(Path(input.encoder.value))
    current_synt = Synthesizer(Path(input.synthesizer.value))
    gan_vocoder.load_model(Path(input.vocoder.value))

    # load file
    if input.upload_audio_file != NULL:
        with open(TEMP_SOURCE_AUDIO, "w+b") as f:
            f.write(input.upload_audio_file.as_bytes())
            f.seek(0)
        wav, sample_rate = librosa.load(TEMP_SOURCE_AUDIO)
    else:
        wav, sample_rate  = librosa.load(input.local_audio_file.value)
        write(TEMP_SOURCE_AUDIO, sample_rate, wav) #Make sure we get the correct wav

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
    return Output(source_file=source_file, result_file=result_file)