from pathlib import Path
from control.toolbox import Toolbox
from utils.argutils import print_args
from utils.modelutils import check_model_paths
import argparse
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Runs the toolbox",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("-d", "--datasets_root", type=Path, help= \
        "Path to the directory containing your datasets. See toolbox/__init__.py for a list of "
        "supported datasets.", default=None)
    parser.add_argument("-vc", "--vc_mode", action="store_true", 
                        help="Voice Conversion Mode(PPG based)")
    parser.add_argument("-e", "--enc_models_dir", type=Path, default=f"data{os.sep}ckpt{os.sep}encoder", 
                        help="Directory containing saved encoder models")
    parser.add_argument("-s", "--syn_models_dir", type=Path, default=f"data{os.sep}ckpt{os.sep}synthesizer", 
                        help="Directory containing saved synthesizer models")
    parser.add_argument("-v", "--voc_models_dir", type=Path, default=f"data{os.sep}ckpt{os.sep}vocoder", 
                        help="Directory containing saved vocoder models")
    parser.add_argument("-ex", "--extractor_models_dir", type=Path, default=f"data{os.sep}ckpt{os.sep}ppg_extractor", 
                        help="Directory containing saved extrator models")
    parser.add_argument("-cv", "--convertor_models_dir", type=Path, default=f"data{os.sep}ckpt{os.sep}ppg2mel", 
                        help="Directory containing saved convert models")
    parser.add_argument("--cpu", action="store_true", help=\
        "If True, processing is done on CPU, even when a GPU is available.")
    parser.add_argument("--seed", type=int, default=None, help=\
        "Optional random number seed value to make toolbox deterministic.")
    parser.add_argument("--no_mp3_support", action="store_true", help=\
        "If True, no mp3 files are allowed.")
    args = parser.parse_args()
    print_args(args, parser)

    if args.cpu:
        # Hide GPUs from Pytorch to force CPU processing
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    del args.cpu

    ## Remind the user to download pretrained models if needed
    check_model_paths(encoder_path=args.enc_models_dir, synthesizer_path=args.syn_models_dir,
                      vocoder_path=args.voc_models_dir)

    # Launch the toolbox
    Toolbox(**vars(args))    
