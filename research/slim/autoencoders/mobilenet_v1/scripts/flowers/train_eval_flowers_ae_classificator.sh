#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
MODEL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_ae_flowers_batch_1
TRAIN_DIR=${MODEL_DIR}/train_ae_classificator
EVAL_DIR=${MODEL_DIR}/eval_ae_classificator
AUTOENCODER_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_ae_flowers_batch_1/train_ae/model.ckpt-1667
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --model_name=mobilenet_v1 \
    --batch_size=1 \
    --num_epoch=18 \
    --save_summaries_secs=1 \
    --learning_rate=0.001
#    --ae_path=${AUTOENCODER_PATH}
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1
