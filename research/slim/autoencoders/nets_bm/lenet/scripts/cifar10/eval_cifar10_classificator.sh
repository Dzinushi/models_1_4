#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/cifar10
CHECKPOINT_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_cifar10_rmsprop_1_epoch/train_classificator/
EVAL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_cifar10_rmsprop_1_epoch/eval_classificator
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=cifar10 \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=test \
    --model_name=lenet
