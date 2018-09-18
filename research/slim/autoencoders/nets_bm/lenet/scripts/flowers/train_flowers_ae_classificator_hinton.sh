#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_sdc_1_custom_grad_flowers_hinton_epoch_1/train_ae_classificator
AUTOENCODER_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_sdc_1_custom_grad_flowers_hinton_epoch_1/train_ae/
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=sgd \
    --model_name=lenet \
    --batch_size=1 \
    --num_epoch=6 \
    --save_summaries_secs=1 \
    --learning_rate=0.0001 \
    --ae_path=${AUTOENCODER_PATH} \
    --activation=sigmoid
