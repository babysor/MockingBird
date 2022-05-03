import sys
import torch
import argparse
import numpy as np
from utils.load_yaml import HpsYaml
from ppg2mel.train.train_linglf02mel_seq2seq_oneshotvc import Solver

# For reproducibility, comment these may speed up training
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    # Arguments
    parser = argparse.ArgumentParser(description=
            'Training PPG2Mel VC model.')
    parser.add_argument('--config', type=str, 
                        help='Path to experiment config, e.g., config/vc.yaml')
    parser.add_argument('--name', default=None, type=str, help='Name for logging.')
    parser.add_argument('--logdir', default='log/', type=str,
                        help='Logging path.', required=False)
    parser.add_argument('--ckpdir', default='ckpt/', type=str,
                        help='Checkpoint path.', required=False)
    parser.add_argument('--outdir', default='result/', type=str,
                        help='Decode output path.', required=False)
    parser.add_argument('--load', default=None, type=str,
                        help='Load pre-trained model (for training only)', required=False)
    parser.add_argument('--warm_start', action='store_true',
                        help='Load model weights only, ignore specified layers.')
    parser.add_argument('--seed', default=0, type=int,
                        help='Random seed for reproducable results.', required=False)
    parser.add_argument('--njobs', default=8, type=int,
                        help='Number of threads for dataloader/decoding.', required=False)
    parser.add_argument('--cpu', action='store_true', help='Disable GPU training.')
    # parser.add_argument('--no-pin', action='store_true',
    #                     help='Disable pin-memory for dataloader')
    parser.add_argument('--no-msg', action='store_true', help='Hide all messages.')
    
    ###

    paras = parser.parse_args()
    setattr(paras, 'gpu', not paras.cpu)
    setattr(paras, 'pin_memory', not paras.no_pin)
    setattr(paras, 'verbose', not paras.no_msg)
    # Make the config dict dot visitable
    config = HpsYaml(paras.config)

    np.random.seed(paras.seed)
    torch.manual_seed(paras.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(paras.seed)

    print(">>> OneShot VC training ...")
    mode = "train"
    solver = Solver(config, paras, mode)
    solver.load_data()
    solver.set_model()
    solver.exec()
    print(">>> Oneshot VC train finished!")
    sys.exit(0)

if __name__ == "__main__":
    main()   
