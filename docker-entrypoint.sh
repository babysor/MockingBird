#!/usr/bin/env bash

if [ -z "$(ls -A /workspace/synthesizer/saved_models)" ] || [ "$FORCE_RETRAIN" = true ] ; then
    /workspace/datasets_download/download.sh
    /workspace/datasets_download/extract.sh
    for DATASET in ${TRAIN_DATASETS}
    do
        if [ "$TRAIN_SKIP_EXISTING" = true ] ; then
            python pre.py /datasets -d ${DATASET} -n $(nproc) --skip_existing
        else
            python pre.py /datasets -d ${DATASET} -n $(nproc)
        fi
    done
    python synthesizer_train.py mandarin /datasets/SV2TTS/synthesizer
fi

python web.py
