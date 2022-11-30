from synthesizer.inference import Synthesizer
from pydantic import BaseModel, Field
from encoder import inference as speacker_encoder
import torch
import os
from pathlib import Path
from enum import Enum
import ppg_extractor as Extractor
import ppg2mel as Convertor
import librosa
from scipy.io.wavfile import write
import re
import numpy as np
from mkgui.base.components.types import FileContent
from vocoder.hifigan import inference as gan_vocoder
from typing import Any, Tuple
import matplotlib.pyplot as plt


# Constants
AUDIO_SAMPLES_DIR = f'samples{os.sep}'
EXT_MODELS_DIRT = f'ppg_extractor{os.sep}saved_models'
CONV_MODELS_DIRT = f'ppg2mel{os.sep}saved_models'
VOC_MODELS_DIRT = f'vocoder{os.sep}saved_models'
TEMP_SOURCE_AUDIO = f'wavs{os.sep}temp_source.wav'
TEMP_TARGET_AUDIO = f'wavs{os.sep}temp_target.wav'
TEMP_RESULT_AUDIO = f'wavs{os.sep}temp_result.wav'

# Load local sample audio as options TODO: load dataset 
if os.path.isdir(AUDIO_SAMPLES_DIR):
    audio_input_selection = Enum('samples', list((file.name, file) for file in Path(AUDIO_SAMPLES_DIR).glob("*.wav")))
# Pre-Load models
if os.path.isdir(EXT_MODELS_DIRT):    
    extractors =  Enum('extractors', list((file.name, file) for file in Path(EXT_MODELS_DIRT).glob("**/*.pt")))
    print("Loaded extractor models: " + str(len(extractors)))
else:
    raise Exception(f"Model folder {EXT_MODELS_DIRT} doesn't exist.")

if os.path.isdir(CONV_MODELS_DIRT):    
    convertors =  Enum('convertors', list((file.name, file) for file in Path(CONV_MODELS_DIRT).glob("**/*.pth")))
    print("Loaded convertor models: " + str(len(convertors)))
else:
    raise Exception(f"Model folder {CONV_MODELS_DIRT} doesn't exist.")

if os.path.isdir(VOC_MODELS_DIRT):    
    vocoders =  Enum('vocoders', list((file.name, file) for file in Path(VOC_MODELS_DIRT).glob("**/*gan*.pt")))
    print("Loaded vocoders models: " + str(len(vocoders)))
else:
    raise Exception(f"Model folder {VOC_MODELS_DIRT} doesn't exist.")

class Input(BaseModel):
    local_audio_file: audio_input_selection = Field(
        ..., alias="输入语音（本地wav）",
        description="选择本地语音文件."
    )
    upload_audio_file: FileContent = Field(default=None, alias="或上传语音",
        description="拖拽或点击上传.", mime_type="audio/wav")
    local_audio_file_target: audio_input_selection = Field(
        ..., alias="目标语音（本地wav）",
        description="选择本地语音文件."
    )
    upload_audio_file_target: FileContent = Field(default=None, alias="或上传目标语音",
        description="拖拽或点击上传.", mime_type="audio/wav")
    extractor: extractors = Field(
        ..., alias="编码模型", 
        description="选择语音编码模型文件."
    )
    convertor: convertors = Field(
        ..., alias="转换模型", 
        description="选择语音转换模型文件."
    )
    vocoder: vocoders = Field(
        ..., alias="语音解码模型", 
        description="选择语音解码模型文件(目前只支持HifiGan类型)."
    )

class AudioEntity(BaseModel):
    content: bytes
    mel: Any

class Output(BaseModel):
    __root__: Tuple[AudioEntity, AudioEntity, AudioEntity]

    def render_output_ui(self, streamlit_app, input) -> None:  # type: ignore
        """Custom output UI.
        If this method is implmeneted, it will be used instead of the default Output UI renderer.
        """
        src, target, result = self.__root__
        
        streamlit_app.subheader("Synthesized Audio")
        streamlit_app.audio(result.content, format="audio/wav")

        fig, ax = plt.subplots()
        ax.imshow(src.mel, aspect="equal", interpolation="none")
        ax.set_title("mel spectrogram(Source Audio)")
        streamlit_app.pyplot(fig)
        fig, ax = plt.subplots()
        ax.imshow(target.mel, aspect="equal", interpolation="none")
        ax.set_title("mel spectrogram(Target Audio)")
        streamlit_app.pyplot(fig)
        fig, ax = plt.subplots()
        ax.imshow(result.mel, aspect="equal", interpolation="none")
        ax.set_title("mel spectrogram(Result Audio)")
        streamlit_app.pyplot(fig)

def convert(input: Input) -> Output:
    """convert(转换)"""
    # load models
    extractor = Extractor.load_model(Path(input.extractor.value))
    convertor = Convertor.load_model(Path(input.convertor.value))
    # current_synt = Synthesizer(Path(input.synthesizer.value))
    gan_vocoder.load_model(Path(input.vocoder.value))

    # load file
    if input.upload_audio_file != None:
        with open(TEMP_SOURCE_AUDIO, "w+b") as f:
            f.write(input.upload_audio_file.as_bytes())
            f.seek(0)
        src_wav, sample_rate = librosa.load(TEMP_SOURCE_AUDIO)
    else:
        src_wav, sample_rate  = librosa.load(input.local_audio_file.value)
        write(TEMP_SOURCE_AUDIO, sample_rate, src_wav) #Make sure we get the correct wav

    if input.upload_audio_file_target != None:
        with open(TEMP_TARGET_AUDIO, "w+b") as f:
            f.write(input.upload_audio_file_target.as_bytes())
            f.seek(0)
        ref_wav, _ = librosa.load(TEMP_TARGET_AUDIO)
    else:
        ref_wav, _  = librosa.load(input.local_audio_file_target.value)
        write(TEMP_TARGET_AUDIO, sample_rate, ref_wav) #Make sure we get the correct wav

    ppg = extractor.extract_from_wav(src_wav)
    # Import necessary dependency of Voice Conversion
    from utils.f0_utils import compute_f0, f02lf0, compute_mean_std, get_converted_lf0uv   
    ref_lf0_mean, ref_lf0_std = compute_mean_std(f02lf0(compute_f0(ref_wav)))
    speacker_encoder.load_model(Path(f"encoder{os.sep}saved_models{os.sep}pretrained_bak_5805000.pt"))
    embed = speacker_encoder.embed_utterance(ref_wav)
    lf0_uv = get_converted_lf0uv(src_wav, ref_lf0_mean, ref_lf0_std, convert=True)
    min_len = min(ppg.shape[1], len(lf0_uv))
    ppg = ppg[:, :min_len]
    lf0_uv = lf0_uv[:min_len]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, mel_pred, att_ws = convertor.inference(
        ppg,
        logf0_uv=torch.from_numpy(lf0_uv).unsqueeze(0).float().to(device),
        spembs=torch.from_numpy(embed).unsqueeze(0).to(device),
    )
    mel_pred= mel_pred.transpose(0, 1)
    breaks = [mel_pred.shape[1]]
    mel_pred= mel_pred.detach().cpu().numpy()

    # synthesize and vocode
    wav, sample_rate = gan_vocoder.infer_waveform(mel_pred)

    # write and output 
    write(TEMP_RESULT_AUDIO, sample_rate, wav) #Make sure we get the correct wav
    with open(TEMP_SOURCE_AUDIO, "rb") as f:
        source_file = f.read()
    with open(TEMP_TARGET_AUDIO, "rb") as f:
        target_file = f.read()
    with open(TEMP_RESULT_AUDIO, "rb") as f:
        result_file = f.read()
    

    return Output(__root__=(AudioEntity(content=source_file, mel=Synthesizer.make_spectrogram(src_wav)), AudioEntity(content=target_file, mel=Synthesizer.make_spectrogram(ref_wav)), AudioEntity(content=result_file, mel=Synthesizer.make_spectrogram(wav))))