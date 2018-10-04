#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
MODEL_NAME=mobilenet_v1
MODEL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/${MODEL_NAME}_ae_flowers_batch_1
TRAIN_DIR=${MODEL_DIR}/train_classificator_pretrain
EVAL_DIR=${MODEL_DIR}/eval_classificator_pretrain
CHECKPOINT_PATH=/home/haaohi/Загрузки/mobilenet_v1_224/mobilenet_v1_1.0_224.ckpt
python3 /media/data/soft/enviroment/models_old/research/slim/autoencoders/train_slim.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=flowers \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_DIR} \
    --optimizer=rmsprop \
    --model_name=${MODEL_NAME} \
    --batch_size=1 \
    --num_epoch=18 \
    --save_summaries_secs=10 \
    --learning_rate=0.001 \
    --trainable_scopes=MobilenetV1/Logits
#    --checkpoint_exclude_scopes=MobilenetV1/Logits \
#    --checkpoint_path=${CHECKPOINT_PATH}
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${TRAIN_DIR} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=validation \
    --model_name=${MODEL_NAME}
