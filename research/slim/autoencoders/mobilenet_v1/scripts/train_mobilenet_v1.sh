#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/cifar10
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_flowers_ae-true/train
AUTOENCODER_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_ae_cifar10/train/model.ckpt-1000
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=mobilenet_v1 \
    --batch_size=50 \
    --max_number_of_steps=3320 \
    --save_summaries_secs=1 \
    --ae_path=${AUTOENCODER_PATH}
