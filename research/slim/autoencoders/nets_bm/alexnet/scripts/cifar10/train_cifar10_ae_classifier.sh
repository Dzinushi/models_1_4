#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/cifar10
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/alexnet_ae_cifar10_batch_1/train_ae_classifier
AUTOENCODER_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/alexnet_ae_cifar10_batch_1/train_ae/model.ckpt-50000
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --model_name=alexnet_v2 \
    --batch_size=1 \
    --num_epoch=2 \
    --save_summaries_secs=1 \
    --learning_rate=0.001 \
    --ae_path=${AUTOENCODER_PATH}
