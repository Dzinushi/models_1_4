#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_flowers_ae-true/train/model.ckpt-6640 \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --eval_dir=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_flowers_ae-true/eval \
    --dataset_split_name=validation \
    --model_name=mobilenet_v1
