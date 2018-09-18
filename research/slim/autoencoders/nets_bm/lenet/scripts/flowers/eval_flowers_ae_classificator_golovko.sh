#!/usr/bin/env bash
DATASET_DIR=/media/w_programs/NN_Database/data/flowers
CHECKPOINT_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_sdc_1_custom_grad_flowers_golovko_epoch_1/train_ae_classificator/
EVAL_DIR=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/lenet_sdc_1_custom_grad_flowers_golovko_epoch_1/eval_ae_classificator
python3 /media/data/soft/enviroment/models_old/research/slim/eval_image_classifier.py \
    --alsologtostderr \
    --checkpoint_path=${CHECKPOINT_PATH} \
    --dataset_dir=${DATASET_DIR} \
    --dataset_name=flowers \
    --eval_dir=${EVAL_DIR} \
    --dataset_split_name=validation \
    --model_name=lenet
