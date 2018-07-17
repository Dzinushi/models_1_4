import tensorflow as tf
from research.slim.autoencoders.alexnet import alexnet_bm

image = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
block_numbers = alexnet_bm.alexnet_v2.block_number
for block in range(block_numbers):
    with tf.Graph().as_default():
        output, end_points = alexnet_bm.alexnet_v2(image, train_block_num=block, sdc_num=1)
        losses = alexnet_bm.alexnet_model_losses(end_points, train_block_num=block, sdc_num=1)
        print(losses)
