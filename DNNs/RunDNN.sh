#!/bin/bash

cd lama

# Check if the "big-lama" folder exists
if [ ! -d "big-lama" ]; then
    echo "big-lama not found. Downloading..."
    curl -LJO https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.zip
    unzip big-lama.zip
else
    echo "big-lama folder already exists. Skipping download."
fi

# Run the prediction script
bash docker/2_predict_with_gpu.sh "$(pwd)/big-lama" "$(pwd)/../in_lama" "$(pwd)/../out_lama" device=cpu

