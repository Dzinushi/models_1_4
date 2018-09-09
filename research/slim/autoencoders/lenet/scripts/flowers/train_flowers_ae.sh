#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_sdc_1_cutom_grad_flowers_batch_1/train_ae
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_ae_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --formulas=golovko \
    --ae_name=lenet_bm \
    --batch_size=1 \
    --max_number_of_steps=50 \
    --save_summaries_secs=16 \
    --learning_rate=0.001 \
    --log_every_n_steps=1
#    --num_epoch=1 \
