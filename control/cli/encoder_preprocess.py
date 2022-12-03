import argparse
from pathlib import Path

from models.encoder.preprocess import (preprocess_aidatatang_200zh,
                                preprocess_librispeech, preprocess_voxceleb1,
                                preprocess_voxceleb2)
from utils.argutils import print_args

if __name__ == "__main__":
    class MyFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass
    
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms and "
                    "writes them to the disk. This will allow you to train the encoder. The "
                    "datasets required are at least one of LibriSpeech, VoxCeleb1, VoxCeleb2, aidatatang_200zh. ",
        formatter_class=MyFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS and VoxCeleb datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms. If left out, "
        "defaults to <datasets_root>/SV2TTS/encoder/")
    parser.add_argument("-d", "--datasets", type=str, 
                        default="librispeech_other,voxceleb1,aidatatang_200zh", help=\
        "Comma-separated list of the name of the datasets you want to preprocess. Only the train "
        "set of these datasets will be used. Possible names: librispeech_other, voxceleb1, "
        "voxceleb2.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to skip existing output files with the same name. Useful if this script was "
        "interrupted.")
    parser.add_argument("--no_trim", action="store_true", help=\
        "Preprocess audio without trimming silences (not recommended).")
    args = parser.parse_args()

    # Verify webrtcvad is available
    if not args.no_trim:
        try:
            import webrtcvad
        except:
            raise ModuleNotFoundError("Package 'webrtcvad' not found. This package enables "
                "noise removal and is recommended. Please install and try again. If installation fails, "
                "use --no_trim to disable this error message.")
    del args.no_trim

    # Process the arguments
    args.datasets = args.datasets.split(",")
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "encoder")
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Preprocess the datasets
    print_args(args, parser)
    preprocess_func = {
        "librispeech_other": preprocess_librispeech,
        "voxceleb1": preprocess_voxceleb1,
        "voxceleb2": preprocess_voxceleb2,
        "aidatatang_200zh": preprocess_aidatatang_200zh,
    }
    args = vars(args)
    for dataset in args.pop("datasets"):
        print("Preprocessing %s" % dataset)
        preprocess_func[dataset](**args)
