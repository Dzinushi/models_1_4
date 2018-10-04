#!/usr/bin/env bash
FORMULAS=hinton
DATASET_DIR=/media/w_programs/Development/Python/tf_autoencoders/datasets/flowers_np_img_size_223_223.sqlite
MODEL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/alexnet_sdc_1_custom_grad_flowers_${FORMULAS}_epoch_1_batch_1
TRAIN_DIR=${MODEL_DIR}/train_ae
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_ae_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --formulas=${FORMULAS} \
    --ae_name=alexnet_bm \
    --batch_size=1 \
    --save_summaries_secs=1 \
    --save_every_step=332 \
    --learning_rate=0.0001 \
    --log_every_n_steps=10 \
    --max_number_of_steps=10
#    --num_epoch=1
