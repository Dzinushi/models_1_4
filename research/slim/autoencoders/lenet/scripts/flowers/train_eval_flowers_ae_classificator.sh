#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
MODEL_NAME=lenet
MODEL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/${MODEL_NAME}_ae_flowers_batch_1
TRAIN_DIR=${MODEL_DIR}/train_ae_classificator
EVAL_DIR=${MODEL_DIR}/eval_ae_classificator
AUTOENCODER_PATH=${MODEL_DIR}/train_ae
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --model_name=${MODEL_NAME} \
    --batch_size=1 \
    --num_epoch=2 \
    --save_summaries_secs=10 \
    --learning_rate=0.0001 \
    --log_every_n_steps=100 \
    --ae_path=${AUTOENCODER_PATH}
#    --checkpoint_path=${TRAIN_DIR}/model.ckpt-49800
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=validation \
    --model_name=${MODEL_NAME}
