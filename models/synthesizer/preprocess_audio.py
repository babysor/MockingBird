import librosa
import numpy as np

from models.encoder import inference as encoder
from utils import logmmse
from models.synthesizer import audio
from pathlib import Path
from pypinyin import Style
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin
import torch
from transformers import Wav2Vec2Processor
from .models.wav2emo import EmotionExtractorModel

class PinyinConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass

pinyin = Pinyin(PinyinConverter()).pinyin


# load model from hub 
device = 'cuda' if torch.cuda.is_available() else "cpu"
model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
processor = Wav2Vec2Processor.from_pretrained(model_name)
model = EmotionExtractorModel.from_pretrained(model_name).to(device)

def extract_emo(
    x: np.ndarray,
    sampling_rate: int,
    embeddings: bool = False,
) -> np.ndarray:
    r"""Predict emotions or extract embeddings from raw audio signal."""
    y = processor(x, sampling_rate=sampling_rate)
    y = y['input_values'][0]
    y = torch.from_numpy(y).to(device)

    # run through model
    with torch.no_grad():
        y = model(y)[0 if embeddings else 1]

    # convert to numpy
    y = y.detach().cpu().numpy()

    return y

def _process_utterance(wav: np.ndarray, text: str, out_dir: Path, basename: str, 
                      mel_fpath: str, wav_fpath: str, hparams, encoder_model_fpath):
    ## FOR REFERENCE:
    # For you not to lose your head if you ever wish to change things here or implement your own
    # synthesizer.
    # - Both the audios and the mel spectrograms are saved as numpy arrays
    # - There is no processing done to the audios that will be saved to disk beyond volume  
    #   normalization (in split_on_silences)
    # - However, pre-emphasis is applied to the audios before computing the mel spectrogram. This
    #   is why we re-apply it on the audio on the side of the vocoder.
    # - Librosa pads the waveform before computing the mel spectrogram. Here, the waveform is saved
    #   without extra padding. This means that you won't have an exact relation between the length
    #   of the wav and of the mel spectrogram. See the vocoder data loader.
        
    # Trim silence
    if hparams.trim_silence:
        if not encoder.is_loaded():
            encoder.load_model(encoder_model_fpath)
        wav = encoder.preprocess_wav(wav, normalize=False, trim_silence=True)
    
    # Skip utterances that are too short
    if len(wav) < hparams.utterance_min_duration * hparams.sample_rate:
        return None
    
    # Compute the mel spectrogram
    mel_spectrogram = audio.melspectrogram(wav, hparams).astype(np.float32)
    mel_frames = mel_spectrogram.shape[1]
    
    # Skip utterances that are too long
    if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
        return None
    # Write the spectrogram, embed and audio to disk
    np.save(mel_fpath, mel_spectrogram.T, allow_pickle=False)
    np.save(wav_fpath, wav, allow_pickle=False)

    # Return a tuple describing this training example
    return wav_fpath.name, mel_fpath.name, "embed-%s.npy" % basename, wav, mel_frames, text
 

def _split_on_silences(wav_fpath, words, hparams):
    # Load the audio waveform
    wav, _ = librosa.load(wav_fpath, sr= hparams.sample_rate)
    wav = librosa.effects.trim(wav, top_db= 40, frame_length=2048, hop_length=1024)[0]
    if hparams.rescale:
        wav = wav / np.abs(wav).max() * hparams.rescaling_max
    # denoise, we may not need it here.
    if len(wav) > hparams.sample_rate*(0.3+0.1):
        noise_wav = np.concatenate([wav[:int(hparams.sample_rate*0.15)],
                                    wav[-int(hparams.sample_rate*0.15):]])
        profile = logmmse.profile_noise(noise_wav, hparams.sample_rate)
        wav = logmmse.denoise(wav, profile, eta=0)

    resp = pinyin(words, style=Style.TONE3)
    res = filter(lambda v : not v.isspace(),map(lambda v: v[0],resp)) 
    res = " ".join(res)

    return wav, res

def preprocess_general(speaker_dir, out_dir: Path, skip_existing: bool, hparams, dict_info, no_alignments: bool, encoder_model_fpath: Path):
    metadata = []
    extensions = ("*.wav", "*.flac", "*.mp3")
    for extension in extensions:
        wav_fpath_list = speaker_dir.glob(extension)
        # Iterate over each wav
        for wav_fpath in wav_fpath_list:
            words = dict_info.get(wav_fpath.name.split(".")[0])
            if not words:
                words = dict_info.get(wav_fpath.name) # try with extension 
                if not words:
                    print(f"No word found in dict_info for {wav_fpath.name}, skip it")
                    continue
            sub_basename = "%s_%02d" % (wav_fpath.name, 0)
            mel_fpath = out_dir.joinpath("mels", f"mel-{sub_basename}.npy")
            wav_fpath = out_dir.joinpath("audio", f"audio-{sub_basename}.npy")
            
            if skip_existing and mel_fpath.exists() and wav_fpath.exists():
                continue
            wav, text = _split_on_silences(wav_fpath, words, hparams)
            result = _process_utterance(wav, text, out_dir, sub_basename, 
                                                False, hparams, encoder_model_fpath) # accelarate
            if result is None:
                continue
            wav_fpath_name, mel_fpath_name, embed_fpath_name, wav, mel_frames, text = result
            metadata.append ((wav_fpath_name, mel_fpath_name, embed_fpath_name, len(wav), mel_frames, text))

    return metadata
