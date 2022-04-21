from __future__ import absolute_import, division, print_function, unicode_literals
import os
import torch
import argparse
from scipy.io.wavfile import write
from vocoder.fregan.utils import AttrDict
import numpy as np
from vocoder.fregan.generator import FreGAN
import json

MAX_WAV_VALUE = 32768.0


def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict


def main(args):
    if args.config is not None:
        with open(args.config) as f:
            data = f.read()
        global h

        json_config = json.loads(data)
        h = AttrDict(json_config)

    torch.manual_seed(h.seed)

    model = FreGAN(h).cuda()
    state_dict_g = load_checkpoint(args.checkpoint_path, 'cuda')
    model.load_state_dict(state_dict_g['generator'])

    model.eval()
    model.remove_weight_norm()

    with torch.no_grad():
        mel = torch.from_numpy(np.load(args.input))
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)
        mel = mel.cuda()
        #zero = torch.full((1, 80, 10), -11.5129).to(mel.device)
        #mel = torch.cat((mel, zero), dim=2)
        hifigan_trace = torch.jit.trace(model, mel)
        print(state_dict_g.keys())
        hifigan_trace.save("{}/hifigan_{}.pt".format(args.out, args.name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default=None, required=True,
                        help="yaml file for config. will use hp_str from checkpoint if not given.")
    parser.add_argument('-p', '--checkpoint_path', type=str, required=True,
                        help="path of checkpoint pt file for evaluation")
    parser.add_argument('-i', '--input', type=str, required=True,
                        help="directory of mel-spectrograms to invert into raw audio. ")
    parser.add_argument('-o', '--out', type=str, required=True,
                        help="path of output pt file")
    parser.add_argument('-n', '--name', type=str, required=True,
                        help="name of the output file")
    args = parser.parse_args()

    main(args)
