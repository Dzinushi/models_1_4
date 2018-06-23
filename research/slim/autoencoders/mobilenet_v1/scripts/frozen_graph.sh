#!/usr/bin/env bash
GRAPH_PATH=/media/data/assets/graph.pb
CHECKPOINT_PATH=/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_flowers_ae-false/train/model.ckpt-9960
python3 /usr/local/lib/python3.5/dist-packages/tensorflow/python/tools/freeze_graph.py \
  --input_graph=${GRAPH_PATH} \
  --input_checkpoint=${CHECKPOINT_PATH} \
  --output_graph=/home/haaohi/TensorFlow/model.ckpt-9960_frozen.pb \
  --output_node_names=softmax
#  --input_binary=true \
