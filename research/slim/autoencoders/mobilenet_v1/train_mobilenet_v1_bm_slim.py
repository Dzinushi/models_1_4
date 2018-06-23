from __future__ import division, print_function, absolute_import

import tensorflow as tf
from preprocessing import inception_preprocessing
from tensorflow.contrib import slim
from datasets import dataset_factory
from autoencoders.mobilenet_v1 import mobilenet_v1_bm

# model_save_path = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/mobilenet_v1_mnist_rmsprop_1_epoch/'
# dataset_dir = '/media/w_programs/NN_Database/data/cifar10'

# dataset = cifar10
# batch_size = 10  # Number of samples in each batch
# epoch_num = 1  # Number of epochs to train the network
# learning_rate = 0.001  # Learning rate

tf.app.flags.DEFINE_integer('max_number_of_steps', None,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'train_dir', '/tmp/tfmodel/',
    'Directory where checkpoints and event logs are written to.')

tf.app.flags.DEFINE_integer(
    'log_every_n_steps', 10,
    'The frequency with which logs are print.')

tf.app.flags.DEFINE_integer(
    'save_summaries_secs', 600,
    'The frequency with which summaries are saved, in seconds.')

tf.app.flags.DEFINE_integer(
    'save_interval_secs', 600,
    'The frequency with which the model is saved, in seconds.')

######################
# Optimization Flags #
######################

tf.app.flags.DEFINE_float(
    'weight_decay', 0.00004, 'The weight decay on the model weights.')

tf.app.flags.DEFINE_string(
    'optimizer', 'rmsprop',
    'The name of the optimizer, one of "adadelta", "adagrad", "adam",'
    '"ftrl", "momentum", "sgd" or "rmsprop".')

tf.app.flags.DEFINE_float(
    'adadelta_rho', 0.95,
    'The decay rate for adadelta.')

tf.app.flags.DEFINE_float(
    'adagrad_initial_accumulator_value', 0.1,
    'Starting value for the AdaGrad accumulators.')

tf.app.flags.DEFINE_float(
    'adam_beta1', 0.9,
    'The exponential decay rate for the 1st moment estimates.')

tf.app.flags.DEFINE_float(
    'adam_beta2', 0.999,
    'The exponential decay rate for the 2nd moment estimates.')

tf.app.flags.DEFINE_float('opt_epsilon', 1.0, 'Epsilon term for the optimizer.')

tf.app.flags.DEFINE_float('ftrl_learning_rate_power', -0.5,
                          'The learning rate power.')

tf.app.flags.DEFINE_float(
    'ftrl_initial_accumulator_value', 0.1,
    'Starting value for the FTRL accumulators.')

tf.app.flags.DEFINE_float(
    'ftrl_l1', 0.0, 'The FTRL l1 regularization strength.')

tf.app.flags.DEFINE_float(
    'ftrl_l2', 0.0, 'The FTRL l2 regularization strength.')

tf.app.flags.DEFINE_float(
    'momentum', 0.9,
    'The momentum for the MomentumOptimizer and RMSPropOptimizer.')

tf.app.flags.DEFINE_float('rmsprop_momentum', 0.9, 'Momentum.')

tf.app.flags.DEFINE_float('rmsprop_decay', 0.9, 'Decay term for RMSProp.')

#######################
# Learning Rate Flags #
#######################

tf.app.flags.DEFINE_string(
    'learning_rate_decay_type',
    'exponential',
    'Specifies how the learning rate is decayed. One of "fixed", "exponential",'
    ' or "polynomial"')

tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')

tf.app.flags.DEFINE_float(
    'end_learning_rate', 0.0001,
    'The minimal end learning rate used by a polynomial decay learning rate.')

tf.app.flags.DEFINE_float(
    'label_smoothing', 0.0, 'The amount of label smoothing.')

tf.app.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')

tf.app.flags.DEFINE_float(
    'num_epochs_per_decay', 2.0,
    'Number of epochs after which learning rate decays.')

tf.app.flags.DEFINE_integer(
    'num_epoch', -1, 'The number of training epoch')

tf.app.flags.DEFINE_integer(
    'batch_size', 50, 'The number of samples in each batch.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', None,
    'The path to a checkpoint from which to fine-tune.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'cifar10', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'train', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

FLAGS = tf.app.flags.FLAGS


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


# def train_batch(sess, train_op, epoch, batch):
#     start_time = time.time()
#     # total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
#     _, c = sess.run([train_op, loss])
#     time_elapsed = time.time() - start_time
#     tf.logging.info(
#         'Epoch={}, batch={}/{}, cost= {:.5f}, ({:.3f} sec/step)'.format((epoch + 1), batch + 1, batch_per_ep, c,
#                                                                         time_elapsed))
#     return c


def _configure_optimizer(learning_rate):
    """Configures the optimizer used for training.

    Args:
      learning_rate: A scalar or `Tensor` learning rate.

    Returns:
      An instance of an optimizer.

    Raises:
      ValueError: if FLAGS.optimizer is not recognized.
    """
    if FLAGS.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=FLAGS.adadelta_rho,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=FLAGS.adagrad_initial_accumulator_value)
    elif FLAGS.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=FLAGS.adam_beta1,
            beta2=FLAGS.adam_beta2,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=FLAGS.ftrl_learning_rate_power,
            initial_accumulator_value=FLAGS.ftrl_initial_accumulator_value,
            l1_regularization_strength=FLAGS.ftrl_l1,
            l2_regularization_strength=FLAGS.ftrl_l2)
    elif FLAGS.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=FLAGS.momentum,
            name='Momentum')
    elif FLAGS.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=FLAGS.rmsprop_decay,
            momentum=FLAGS.rmsprop_momentum,
            epsilon=FLAGS.opt_epsilon)
    elif FLAGS.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', FLAGS.optimizer)
    return optimizer


def init_fn():
    """Returns a function run by the chief worker to warm-start the training.

      Note that the init_fn is only run when initializing the model during the very
      first global step.

      Returns:
        An init function run by the supervisor.
      """

    if FLAGS.checkpoint_path is None:
        return None

        # Warn the user if a checkpoint exists in the train_dir. Then we'll be
        # ignoring the checkpoint anyway.
    if tf.train.latest_checkpoint(FLAGS.train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % FLAGS.train_dir)
        return None

    exclusions = []
    if FLAGS.checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in FLAGS.checkpoint_exclude_scopes.split(',')]

    # TODO(sguada) variables.filter_variables()
    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
        checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Fine-tuning from %s' % checkpoint_path)

    return slim.assign_from_checkpoint_fn(
        checkpoint_path,
        variables_to_restore,
        ignore_missing_vars=FLAGS.ignore_missing_vars)


def main(_):
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)

        ######################
        # Select the dataset #
        ######################
        dataset = dataset_factory.get_dataset(
            FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

        image_size = mobilenet_v1_bm.mobilenet_v1.default_image_size
        images, labels = load_batch(dataset, FLAGS.batch_size, image_size, image_size, is_training=True)

        # calculate the number of batches per epoch
        batch_per_ep = dataset.num_samples // FLAGS.batch_size

        # Get model data
        ae_outputs, end_points = mobilenet_v1_bm.mobilenet_v1(images, sdc_num=1, channel=3)

        # =================================================================================================
        # LOSS
        # =================================================================================================
        # loss_op = tf.reduce_sum([slim.losses.softmax_cross_entropy(end_points['Conv2d_0_dconv2d'],
        #                                                            end_points['input'],
        #                                                            label_smoothing=0.0, weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_1_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_0'],
        #                                                            label_smoothing=0.0, weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_1_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_1_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_2_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_1_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_2_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_2_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_3_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_2_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_3_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_3_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_4_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_3_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_4_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_4_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_5_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_4_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_5_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_5_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_6_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_5_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_6_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_6_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_7_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_6_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_7_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_7_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_8_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_7_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_8_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_8_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_9_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_8_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_9_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_9_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_10_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_9_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_10_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_10_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_11_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_10_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_11_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_11_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_12_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_11_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_12_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_12_depthwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_13_depthwise_dconv2d'],
        #                                                            end_points['Conv2d_12_pointwise'],
        #                                                            label_smoothing=0.0,
        #                                                            weights=0.4),
        #                          slim.losses.softmax_cross_entropy(end_points['Conv2d_13_pointwise_dconv2d'],
        #                                                            end_points['Conv2d_13_depthwise'],
        #                                                            label_smoothing=0.0, weights=0.4)])

        loss_op = tf.reduce_mean(tf.square(end_points['input'] - end_points['Conv2d_0_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_0'] - end_points['Conv2d_1_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_1_depthwise'] - end_points['Conv2d_1_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_1_pointwise'] - end_points['Conv2d_2_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_2_depthwise'] - end_points['Conv2d_2_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_2_pointwise'] - end_points['Conv2d_3_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_3_depthwise'] - end_points['Conv2d_3_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_3_pointwise'] - end_points['Conv2d_4_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_4_depthwise'] - end_points['Conv2d_4_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_4_pointwise'] - end_points['Conv2d_5_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_5_depthwise'] - end_points['Conv2d_5_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_5_pointwise'] - end_points['Conv2d_6_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_6_depthwise'] - end_points['Conv2d_6_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_6_pointwise'] - end_points['Conv2d_7_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_7_depthwise'] - end_points['Conv2d_7_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_7_pointwise'] - end_points['Conv2d_8_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_8_depthwise'] - end_points['Conv2d_8_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_8_pointwise'] - end_points['Conv2d_9_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_9_depthwise'] - end_points['Conv2d_9_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_9_pointwise'] - end_points['Conv2d_10_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_10_depthwise'] - end_points['Conv2d_10_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_10_pointwise'] - end_points['Conv2d_11_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_11_depthwise'] - end_points['Conv2d_11_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_11_pointwise'] - end_points['Conv2d_12_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_12_depthwise'] - end_points['Conv2d_12_pointwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_12_pointwise'] - end_points['Conv2d_13_depthwise_dconv2d'])) + \
               tf.reduce_mean(tf.square(end_points['Conv2d_13_depthwise'] - end_points['Conv2d_13_pointwise_dconv2d']))
        optimizer = _configure_optimizer(FLAGS.learning_rate)
        # optimizer = tf.train.RMSPropOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9)
        train_op = slim.learning.create_train_op(loss_op, optimizer)

        # init = init_fn()

        # Gather initial summaries.
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

        # Add summaries for end_points.
        for end_point in end_points:
            x = end_points[end_point]
            summaries.add(tf.summary.histogram('activations/' + end_point, x))
            summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                            tf.nn.zero_fraction(x)))

        # Add summaries for losses.
        for loss in tf.get_collection(tf.GraphKeys.LOSSES):
            summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

        # Add summaries for variables.
        for variable in slim.get_model_variables():
            summaries.add(tf.summary.histogram(variable.op.name, variable))

        # Merge all summaries together.
        summary_op = tf.summary.merge(list(summaries), name='summary_op')

        if FLAGS.num_epoch == -1:
            number_of_steps = FLAGS.max_number_of_steps
        else:
            number_of_steps = batch_per_ep * FLAGS.num_epoch

        slim.learning.train(train_op,
                            master=FLAGS.master,
                            logdir=FLAGS.train_dir,
                            number_of_steps=number_of_steps,
                            init_fn=init_fn(),
                            summary_op=summary_op,
                            log_every_n_steps=FLAGS.log_every_n_steps,
                            save_summaries_secs=FLAGS.save_summaries_secs,
                            save_interval_secs=FLAGS.save_interval_secs)


if __name__ == '__main__':
    tf.app.run()
