from models.synthesizer.hparams import hparams
from models.synthesizer.train import train
from utils.argutils import print_args
import argparse

def new_train():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("syn_dir", type=str, default=argparse.SUPPRESS, help= \
        "Path to the synthesizer directory that contains the ground truth mel spectrograms, "
        "the wavs and the embeds.")
    parser.add_argument("-m", "--models_dir", type=str, default=f"data/ckpt/synthesizer/", help=\
        "Path to the output directory that will contain the saved model weights and the logs.")
    parser.add_argument("-s", "--save_every", type=int, default=1000, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=25000, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-l", "--log_every", type=int, default=200, help= \
        "Number of steps between summary the training info in tensorboard")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model and restart from scratch.")
    parser.add_argument("--hparams", default="",
                        help="Hyperparameter overrides as a comma-separated list of name=value "
							 "pairs")
    args, _ = parser.parse_known_args()
    print_args(args, parser)

    args.hparams = hparams.parse(args.hparams)

    # Run the training
    train(**vars(args))


if __name__ == "__main__":
    new_train()