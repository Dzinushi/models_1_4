#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_ae_flowers_batch_10/train_classificator
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --model_name=lenet \
    --batch_size=10 \
    --num_epoch=10 \
    --save_summaries_secs=1 \
    --learning_rate=0.001
