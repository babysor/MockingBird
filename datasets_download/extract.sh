#!/usr/bin/env bash

set -Eeuo pipefail

tar xvz --directory /datasets/ -f /datasets/download/aidatatang_200zh.tgz aidatatang_200zh/corpus/train/
cd /datasets/aidatatang_200zh/corpus/train/
cat *.tar.gz | tar zxvf - -i
rm -f *.tar.gz

mkdir -p /datasets/magicdata
tar xvz --directory /datasets/magicdata -f /datasets/download/magicdata.tgz train/

mkdir -p /datasets/aishell3
tar xvz --directory /datasets/aishell3 -f /datasets/download/aishell3.tgz train/

tar xvz --directory /datasets/ -f /datasets/download/data_aishell.tgz
cd /datasets/data_aishell/wav/
cat *.tar.gz | tar zxvf - -i
rm -f *.tar.gz
