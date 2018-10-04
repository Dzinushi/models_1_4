#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/Development/Python/tf_autoencoders/datasets/flowers_np_28_28_norm.sqlite
TRAIN_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/flowers/hinton/lenet_sdc_1/cgrad_ep_10_batch_10_relu/train_ae
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/tests/test_train_ae_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --formulas=hinton \
    --activation=relu \
    --ae_name=lenet_bm \
    --batch_size=10 \
    --learning_rate=0.0001 \
    --log_every_n_steps=10 \
    --gpu_memory_fraction=0.5 \
    --image_test_count=1
#    --max_number_of_steps=25
#    --save_every_step=1000
