#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
MODEL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/alexnet_ae_flowers_batch_1
TRAIN_DIR=${MODEL_DIR}/train__ae
EVAL_DIR=${MODEL_DIR}/eval_ae
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_ae_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --ae_name=alexnet_bm \
    --batch_size=1 \
    --num_epoch=25 \
    --save_summaries_secs=10 \
    --learning_rate=0.001
