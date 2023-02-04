import argparse

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
    if paras.type == "vits":
        from models.synthesizer.train_vits import new_train
        new_train()

if __name__ == "__main__":
    main()   
