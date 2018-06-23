#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/cifar10
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_ae_cifar10_batch_10/train
CHECKPOINT=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_ae_cifar10/train/model.ckpt-10
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/mobilenet_v1/train_mobilenet_v1_bm_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=cifar10 \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --batch_size=10 \
    --optimizer=rmsprop \
    --log_every_n_steps=1 \
    --learning_rate=0.001 \
    --num_epoch=1
#    --save_summaries_secs=20
#    --max_number_of_steps=5000 \
#    --checkpoint_path=${CHECKPOINT}
