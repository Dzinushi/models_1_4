import tensorflow as tf
from tensorflow.contrib import slim
from datasets import flowers
from preprocessing import inception_preprocessing
from nets import mobilenet_v1
import math

checkpoint_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_flowers/'
dataset_dir = '/media/w_programs/NN_Database/data/flowers/'
eval_dir = '/media/w_programs/NN_Database/data/flowers/eval/'
num_batches = 10
batch_size = num_batches
max_num_batches = None

# if not FLAGS.dataset_dir:
#     raise ValueError('You must supply the dataset directory with --dataset_dir')


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


tf.logging.set_verbosity(tf.logging.INFO)
with tf.Graph().as_default():
    tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = flowers.get_split('validation', dataset_dir)

    images, labels = load_batch(dataset, batch_size, is_training=False)

    ####################
    # Select the model #
    ####################
    logits, _ = mobilenet_v1.mobilenet_v1(images, num_classes=dataset.num_classes, is_training=False)

    ####################
    # Define the model #
    ####################
    # logits, _ = network_fn(images)

    # if FLAGS.moving_average_decay:
    #     variable_averages = tf.train.ExponentialMovingAverage(
    #         FLAGS.moving_average_decay, tf_global_step)
    #     variables_to_restore = variable_averages.variables_to_restore(
    #         slim.get_model_variables())
    #     variables_to_restore[tf_global_step.op.name] = tf_global_step
    # else:
    #     variables_to_restore = slim.get_variables_to_restore()

    variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
        'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
        'Recall_5': slim.metrics.streaming_recall_at_k(
            logits, labels, 5),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
        summary_name = 'eval/%s' % name
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
        tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if max_num_batches is not None:
        num_batches = max_num_batches
    else:
        # This ensures that we make a single pass over all of the data.
        num_batches = math.ceil(dataset.num_samples / float(batch_size))

    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    else:
        checkpoint_path = checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master='master',
        checkpoint_path=checkpoint_path,
        logdir=eval_dir,
        num_evals=num_batches,
        eval_op=list(names_to_updates.values()),
        variables_to_restore=variables_to_restore)
