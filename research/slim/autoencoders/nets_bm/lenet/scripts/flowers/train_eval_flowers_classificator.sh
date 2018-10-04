#!/usr/bin/env bash
MODEL_FOLDER=flowers/no_autoencoder/lenet_sdc_1/cgrad_ep_10_batch_10_relu
DATASET_DIR=/media/w_programs/Development/Python/tf_autoencoders/datasets/flowers_np_28_28_norm.sqlite
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/${MODEL_FOLDER}/train_ae_classificator/
EVAL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/${MODEL_FOLDER}/eval_ae_classificator
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_mts.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=sgd \
    --model_name=lenet \
    --batch_size=10 \
    --num_epoch=20 \
    --save_summaries_secs=1 \
    --learning_rate=0.0001 \
    --activation=relu
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/eval_mts.py \
    --dataset_dir=${DATASET_DIR} \
    --eval_dir=${TRAIN_DIR} \
    --model_name=lenet
