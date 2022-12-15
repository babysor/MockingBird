#!/usr/bin/env bash

if [ -z "$(ls -A /workspace/synthesizer/saved_models)" ] || [ "$FORCE_RETRAIN" = true ] ; then
    /workspace/datasets_download/download.sh
    /workspace/datasets_download/extract.sh
    for DATASET in ${TRAIN_DATASETS}
    do
        python pre.py /datasets -d ${DATASET} -n $(nproc)
    done
    python synthesizer_train.py mandarin /datasets/SV2TTS/synthesizer
fi

python web.py
