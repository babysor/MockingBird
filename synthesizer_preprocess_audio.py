from synthesizer.preprocess import preprocess_dataset
from synthesizer.hparams import hparams
from utils.argutils import print_args
from pathlib import Path
import argparse


recognized_datasets = [
    "aidatatang_200zh",
    "magicdata",
    "aishell3"
]

if __name__ == "__main__":
    print("This method is deprecaded and will not be longer supported, please use 'pre.py'")
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your LibriSpeech/TTS datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=None, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted.")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--no_trim", action="store_true", help=\
        "Preprocess audio without trimming silences (not recommended).")
    parser.add_argument("--no_alignments", action="store_true", help=\
        "Use this option when dataset does not include alignments\
        (these are used to split long audio files into sub-utterances.)")
    parser.add_argument("--dataset", type=str, default="aidatatang_200zh", help=\
        "Name of the dataset to process, allowing values: magicdata, aidatatang_200zh.")
    args = parser.parse_args()

    # Process the arguments
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("SV2TTS", "synthesizer")
    assert args.dataset in recognized_datasets, 'is not supported, please vote for it in https://github.com/babysor/MockingBird/issues/10'
    # Create directories
    assert args.datasets_root.exists()
    args.out_dir.mkdir(exist_ok=True, parents=True)

    # Verify webrtcvad is available
    if not args.no_trim:
        try:
            import webrtcvad
        except:
            raise ModuleNotFoundError("Package 'webrtcvad' not found. This package enables "
                "noise removal and is recommended. Please install and try again. If installation fails, "
                "use --no_trim to disable this error message.")
    del args.no_trim

    # Preprocess the dataset
    print_args(args, parser)
    args.hparams = hparams.parse(args.hparams)

    preprocess_dataset(**vars(args))