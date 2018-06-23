#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_flowers_ae-false-pretrained/train/model.ckpt-3320 \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --eval_dir=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_flowers_ae-false-pretrained/eval \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1
