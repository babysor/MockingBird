from models.synthesizer.preprocess import create_embeddings, preprocess_dataset, create_emo
from models.synthesizer.hparams import hparams
from pathlib import Path
import argparse

recognized_datasets = [
    "aidatatang_200zh",
    "aidatatang_200zh_s",
    "magicdata",
    "aishell3",
    "data_aishell"
]

#TODO: add for emotional data 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocesses audio files from datasets, encodes them as mel spectrograms "
                    "and writes them to  the disk. Audio files are also saved, to be used by the "
                    "vocoder for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("datasets_root", type=Path, help=\
        "Path to the directory containing your datasets.")
    parser.add_argument("-o", "--out_dir", type=Path, default=argparse.SUPPRESS, help=\
        "Path to the output directory that will contain the mel spectrograms, the audios and the "
        "embeds. Defaults to <datasets_root>/SV2TTS/synthesizer/")
    parser.add_argument("-n", "--n_processes", type=int, default=1, help=\
        "Number of processes in parallel.")
    parser.add_argument("-s", "--skip_existing", action="store_true", help=\
        "Whether to overwrite existing files with the same name. Useful if the preprocessing was "
        "interrupted. ")
    parser.add_argument("--hparams", type=str, default="", help=\
        "Hyperparameter overrides as a comma-separated list of name-value pairs")
    parser.add_argument("--no_trim", action="store_true", help=\
        "Preprocess audio without trimming silences (not recommended).")
    parser.add_argument("--no_alignments", action="store_true", help=\
        "Use this option when dataset does not include alignments\
        (these are used to split long audio files into sub-utterances.)")
    parser.add_argument("-d", "--dataset", type=str, default="aidatatang_200zh", help=\
        "Name of the dataset to process, allowing values: magicdata, aidatatang_200zh, aishell3, data_aishell.")
    parser.add_argument("-e", "--encoder_model_fpath", type=Path, default="data/ckpt/encoder/pretrained.pt", help=\
        "Path your trained encoder model.")
    parser.add_argument("-ne", "--n_processes_embed", type=int, default=1, help=\
        "Number of processes in parallel.An encoder is created for each, so you may need to lower "
        "this value on GPUs with low memory. Set it to 1 if CUDA is unhappy")
    parser.add_argument("-ee","--emotion_extract", action="store_true", help=\
        "Preprocess audio to extract emotional numpy (for emotional vits).")
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
    encoder_model_fpath = args.encoder_model_fpath
    del args.no_trim
   
    args.hparams = hparams.parse(args.hparams)
    n_processes_embed = args.n_processes_embed
    del args.n_processes_embed
    preprocess_dataset(**vars(args))
    
    create_embeddings(synthesizer_root=args.out_dir, n_processes=n_processes_embed, encoder_model_fpath=encoder_model_fpath, skip_existing=args.skip_existing)
    
    if args.emotion_extract:
        create_emo(synthesizer_root=args.out_dir, n_processes=n_processes_embed, skip_existing=args.skip_existing, hparams=args.hparams)
