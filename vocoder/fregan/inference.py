from __future__ import absolute_import, division, print_function, unicode_literals

import os
import json
import torch
from scipy.io.wavfile import write
from vocoder.hifigan.env import AttrDict
from vocoder.hifigan.meldataset import mel_spectrogram, MAX_WAV_VALUE, load_wav
from vocoder.fregan.generator import FreGAN
import soundfile as sf


generator = None       # type: FreGAN
_device = None


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def load_model(weights_fpath, verbose=True):
    global generator, _device

    if verbose:
        print("Building fregan")

    with open("./vocoder/fregan/config.json") as f:
        data = f.read()
    json_config = json.loads(data)
    h = AttrDict(json_config)
    torch.manual_seed(h.seed)

    if torch.cuda.is_available():
        # _model = _model.cuda()
        _device = torch.device('cuda')
    else:
        _device = torch.device('cpu')

    generator = FreGAN(h).to(_device)
    state_dict_g = load_checkpoint(
        weights_fpath, _device
    )
    generator.load_state_dict(state_dict_g['generator'])
    generator.eval()
    generator.remove_weight_norm()


def is_loaded():
    return generator is not None


def infer_waveform(mel, progress_callback=None):

    if generator is None:
        raise Exception("Please load fre-gan in memory before using it")

    mel = torch.FloatTensor(mel).to(_device)
    mel = mel.unsqueeze(0)

    with torch.no_grad():
        y_g_hat = generator(mel)
        audio = y_g_hat.squeeze()
    audio = audio.cpu().numpy()

    return audio

