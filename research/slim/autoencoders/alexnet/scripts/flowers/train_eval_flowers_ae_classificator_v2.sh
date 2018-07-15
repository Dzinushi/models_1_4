#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
MODEL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/alexnet_ae_flowers_batch_1
TRAIN_DIR=${MODEL_DIR}/train_ae_classificator_from_ae_cifar10
EVAL_DIR=${MODEL_DIR}/eval_ae_classificator_from_ae_cifar10
AUTOENCODER_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/alexnet_ae_cifar10_batch_1/train_ae/model.ckpt-50000
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --model_name=alexnet_v2 \
    --batch_size=1 \
    --max_number_of_steps=53320 \
    --save_summaries_secs=1 \
    --learning_rate=0.001 \
    --ae_path=${AUTOENCODER_PATH}
#    --num_epoch=1
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=validation \
    --model_name=alexnet_v2
