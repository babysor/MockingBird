from __future__ import absolute_import, division, print_function, unicode_literals

import glob
import os
import argparse
import json
import torch
import numpy as np
from scipy.io.wavfile import write
from env import AttrDict
from meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from models import Generator
import soundfile as sf


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


h = None
device = None


with open("config_16k_.json") as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
torch.manual_seed(h.seed)
device = torch.device("cpu")


generator = Generator(h).to(device)
state_dict_g = load_checkpoint("../../../TTS/Vocoder/outputs/hifi-gan/models/g_00930000", device)
generator.load_state_dict(state_dict_g['generator'])
generator.eval()
generator.remove_weight_norm()


mel = np.load("./mel-T0055G0184S0349.wav_00.npy")
mel = torch.FloatTensor(mel.T).to(device)
mel = mel.unsqueeze(0)


with torch.no_grad():
    y_g_hat = generator(mel)
    audio = y_g_hat.squeeze()


audio = audio.cpu().numpy()
sf.write("a.wav", audio, samplerate=16000)


# import IPython.display as ipd
# ipd.Audio(audio, rate=16000)