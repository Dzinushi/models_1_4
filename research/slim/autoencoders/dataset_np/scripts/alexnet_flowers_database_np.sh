#!/usr/bin/env bash
SIZE=224
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
SAVE_DATASET_DIR=/media/w_programs/Development/Python/tf_autoencoders/datasets/flowers_np_img_size_${SIZE}_${SIZE}_bipolar.sqlite
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/dataset_np/flowers_to_np_database.py \
    --path=${DATASET_DIR} \
    --save_path=${SAVE_DATASET_DIR} \
    --height=${SIZE} \
    --width=${SIZE} \
    --bipolar=True
