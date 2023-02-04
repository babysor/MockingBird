#!/usr/bin/env bash

set -Eeuo pipefail

aria2c -x 10 --disable-ipv6 --input-file /workspace/datasets_download/${DATASET_MIRROR}.txt --dir /datasets --continue

echo "Verifying sha256sum..."
parallel --will-cite -a /workspace/datasets_download/datasets.sha256sum "echo -n {} | sha256sum -c"
