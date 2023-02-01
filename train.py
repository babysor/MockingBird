import sys
import torch
import argparse
import numpy as np
from utils.load_yaml import HpsYaml
from models.ppg2mel.train.train_linglf02mel_seq2seq_oneshotvc import Solver

def main():
    # Arguments
    preparser = argparse.ArgumentParser(description=
            'Training model.')
    preparser.add_argument('--type', type=str, 
                        help='type of training ')

    ###
    paras, _ = preparser.parse_known_args()
    if paras.type == "synth":
        from control.cli.synthesizer_train import new_train
        new_train()

if __name__ == "__main__":
    main()   
