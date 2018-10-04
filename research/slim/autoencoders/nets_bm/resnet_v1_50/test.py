from nets import mobilenet_v1
import tensorflow as tf
from autoencoders.lenet import lenet_bm
from autoencoders import ae_factory
from tensorflow.contrib import slim
from preprocessing import inception_preprocessing
from datasets import dataset_factory

# model, end_points = mobilenet_v1.mobilenet_v1(inputs=tf.placeholder(tf.float32, shape=(1, 224, 224, 1)),
#                                               num_classes=5)
# for end_point in end_points:
#     print(end_points[end_point])

ae_name = 'lenet_bm'
batch_size = 10
sdc_num = 1
dataset_name = 'flowers'
dataset_split_name = 'train'
dataset_dir = '/media/w_programs/NN_Database/data/flowers'


def load_batch(dataset, batch_size, height=224, width=224, is_training=True):
    ######################
    # Set provider train #
    ######################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=4,
        common_queue_capacity=20 * batch_size,
        common_queue_min=10 * batch_size)
    [image, label] = provider.get(['image', 'label'])

    image = inception_preprocessing.preprocess_image(image, height, width, is_training=True)

    images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size)
    labels = slim.one_hot_encoding(labels, dataset.num_classes)

    return images, labels


dataset = dataset_factory.get_dataset(
    dataset_name, dataset_split_name, dataset_dir)

# model, end_points = lenet_bm.lenet_bm(inputs=tf.placeholder(tf.float32, shape=(1, 28, 28, 3)))
# list_losses = lenet_bm.get_loss_layer_names(end_points)
# for loss in list_losses:
#     print(list_losses[loss])


############################
# Select autoencoder model #
############################
autoencoder_fn, autoencoder_loss_map_fn = ae_factory.get_ae_fn(ae_name)

image_size = autoencoder_fn.default_image_size
images, labels = load_batch(dataset, batch_size, image_size, image_size, is_training=True)

# calculate the number of batches per epoch
batch_per_ep = dataset.num_samples // batch_size

# Get model data
# ae_outputs, end_points = mobilenet_v1_bm.mobilenet_v1(images, sdc_num=1, channel=3)
ae_outputs, end_points = autoencoder_fn(images, sdc_num=sdc_num)

# =================================================================================================
# LOSS
# =================================================================================================

loss_map = autoencoder_loss_map_fn(end_points, sdc_num=sdc_num)
loss_list = []

for index in range(len(loss_map)):
    tf.losses.mean_squared_error(labels=loss_map[index]['input'],
                                 predictions=loss_map[index]['output'])
    print(loss_map[index])
