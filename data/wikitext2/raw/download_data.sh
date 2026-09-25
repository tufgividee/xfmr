#!/bin/bash

set -euo pipefail

wget https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip

for f in *.zip; do unzip -j "$f" && rm "$f"; done
