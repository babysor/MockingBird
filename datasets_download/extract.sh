#!/usr/bin/env bash

set -Eeuo pipefail

mkdir -p /datasets/aidatatang_200zh
if [ -z "$(ls -A /datasets/aidatatang_200zh)" ] ; then
    tar xvz --directory /datasets/ -f /datasets/download/aidatatang_200zh.tgz --exclude 'aidatatang_200zh/corpus/dev/*' --exclude 'aidatatang_200zh/corpus/test/*'
    cd /datasets/aidatatang_200zh/corpus/train/
    cat *.tar.gz | tar zxvf - -i
    rm -f *.tar.gz
fi

mkdir -p /datasets/magicdata
if [ -z "$(ls -A /datasets/magicdata)" ] ; then
    tar xvz --directory /datasets/magicdata -f /datasets/download/magicdata.tgz train/
fi

mkdir -p /datasets/aishell3
if [ -z "$(ls -A /datasets/aishell3)" ] ; then
    tar xvz --directory /datasets/aishell3 -f /datasets/download/aishell3.tgz train/
fi

mkdir -p /datasets/data_aishell
if [ -z "$(ls -A /datasets/data_aishell)" ] ; then
    tar xvz --directory /datasets/ -f /datasets/download/data_aishell.tgz
    cd /datasets/data_aishell/wav/
    cat *.tar.gz | tar zxvf - -i --exclude 'dev/*' --exclude 'test/*'
    rm -f *.tar.gz
fi
