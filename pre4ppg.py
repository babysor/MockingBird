from pathlib import Path
import argparse

from ppg2mel.preprocess import preprocess_dataset
from pathlib import Path
import argparse

recognized_datasets = [
    "aidatatang_200zh",
    "aidatatang_200zh_s", #      sample 
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, to be used by the "
                    "ppg2mel model for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your datasets.")
    parser.add_argument("-d", "--dataset", type=str, default="aidatatang_200zh", help=\
        "Name of the dataset to process, allowing values: aidatatang_200zh.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/PPGVC/ppg2mel/")
    parser.add_argument("-n", "--n_processes", type=int, default=8, help=\
        "Number of processes in parallel.")
    # parser.add_argument("-s", "--skip_existing", action="store_true", help=\
    #     "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
    #     "interrupted. ")
    # parser.add_argument("--hparams", type=str, default="", help=\
    #     "Hyperparameter overrides as a comma-separated list of name-value pairs")
    # parser.add_argument("--no_trim", action="store_true", help=\
    #     "Preprocess audio without trimming silences (not recommended).")
    parser.add_argument("-pf", "--ppg_encoder_model_fpath", type=Path, default="ppg_extractor/saved_models/24epoch.pt", help=\
        "Path your trained ppg encoder model.")
    parser.add_argument("-sf", "--speaker_encoder_model", type=Path, default="encoder/saved_models/pretrained_bak_5805000.pt", help=\
        "Path your trained speaker encoder model.")
    args = parser.parse_args()

    assert args.dataset in recognized_datasets, 'is not supported, file a issue to propose a new one'

    # Create directories
    assert args.datasets_root.exists()
    if not hasattr(args, "out_dir"):
        args.out_dir = args.datasets_root.joinpath("PPGVC", "ppg2mel")
    args.out_dir.mkdir(exist_ok=True, parents=True)

    preprocess_dataset(**vars(args)) 
