#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/cifar10
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_cifar10_ae-false-pretrained/train
CHECKPOINT_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_pretrained/mobilenet_v1_1.0_224.ckpt
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --model_name=mobilenet_v1 \
    --batch_size=50 \
    --max_number_of_steps=1000 \
    --save_summaries_secs=1 \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --checkpoint_exclude_scopes=MobilenetV1/Logits
