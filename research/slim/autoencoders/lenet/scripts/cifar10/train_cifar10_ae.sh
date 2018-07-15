#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/cifar10
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_ae_cifar10_batch_10/train_ae
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_ae_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --ae_name=lenet_bm \
    --batch_size=10 \
    --num_epoch=1 \
    --save_summaries_secs=600 \
    --learning_rate=0.001
