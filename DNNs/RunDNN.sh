#!/bin/bash

cd lama
bash docker/2_predict_with_gpu.sh $(pwd)/big-lama $(pwd)/../in_lama $(pwd)/../out_lama device=cpu
