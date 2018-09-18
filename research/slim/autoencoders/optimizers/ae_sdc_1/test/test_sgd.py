import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from autoencoders.optimizers.ae_sdc_1.sgd import GradientDescentOptimizerSDC1 as sgd_custom
from autoencoders.optimizers.ae_sdc_1.gradient import CustomGradientSDC1 as gradient_custom_cpu
from autoencoders.optimizers.ae_sdc_1.gradient_old import GradientSDC1 as gradient_custom_old_cpu
from tensorflow.contrib import slim
from autoencoders.optimizers.optimizer_utils import Formulas
from timeit import default_timer as timer

max_step = 1000
stride = 1
padding = 'VALID'
formulas = Formulas.hinton
activation_fn = tf.nn.tanh
gradient_custom = gradient_custom_cpu
alpha = 0.001
log_step = 10


# WARNING! first layer must have name 'input'. Recovery layer must have prefix 'recovery'.
def model(input):
    with tf.variable_scope('Model'):
        end_point = {}
        end_point['input_sdc_0'] = net = input
        end_point['output_sdc_0'] = net = slim.conv2d(net, 5, [5, 5], stride=stride, padding=padding,
                                                      activation_fn=activation_fn,
                                                      scope='conv1')
        end_point['input_sdc_1'] = net = slim.conv2d_transpose(net, 3, [5, 5], stride=stride,
                                                               padding=padding, scope='input_recovery')
        end_point['output_sdc_1'] = net = slim.conv2d(net, 5, [5, 5], reuse=True, stride=stride, padding=padding,
                                                      activation_fn=activation_fn,
                                                      scope='conv1')
        return net, end_point


def model_fc(input):
    with tf.variable_scope('Model'):
        end_point = {}
        end_point['input_sdc_0'] = net = input
        end_point['conv1_sdc_0'] = net = slim.conv2d(net, 3, [5, 5], stride=stride, padding='SAME', trainable=False,
                                                     activation_fn=tf.nn.relu, scope='conv1')
        end_point['input_sdc_0'] = net = slim.flatten(net, scope='flatten')
        end_point['output_sdc_0'] = net = slim.fully_connected(net, 2, scope='fc1')
        end_point['input_sdc_1'] = net = slim.fully_connected(net, 4, scope='flatten_recovery')
        end_point['output_sdc_1'] = net = slim.fully_connected(net, 2, reuse=True, scope='fc1')

        return net, end_point


def print_matrix(matrix, name):
    matrix_np = np.array(matrix)
    if len(matrix_np.shape) != 1:
        matrix_np = np.squeeze(matrix_np)
    shape = matrix_np.shape
    if len(shape) == 1:
        print('{} = np.array(['.format(name))
        for x in range(shape[0]):
            smb = ','
            if x + 1 == shape[0]:
                smb = ''
            print('{}{}'.format(matrix_np[x], smb))
        print('])')
    elif len(shape) == 2:
        print(name + ' = np.array([')
        for x in range(shape[0]):
            print('[')
            for y in range(shape[1]):
                smb = ','
                if y + 1 == shape[1]:
                    smb = ''
                print('{}{}'.format(matrix_np[x][y], smb))
            print('],')
        print('])')
    elif len(shape) == 3:
        print(name + ' = np.array([')
        for x in range(shape[0]):
            print('[')
            for y in range(shape[1]):
                print('[')
                for z in range(shape[2]):
                    smb = ','
                    if z + 1 == shape[2]:
                        smb = ''
                    print('{}{}'.format(matrix_np[x][y][z], smb))
                print('],')
            print('],')
        print('])')


def log(sess):
    print_matrix(sess.run(input), 'input_sdc_0')
    print_matrix(sess.run(end_points['input_sdc_1']), 'input_sdc_1')
    print_matrix(sess.run(end_points['output_sdc_0']), 'output_sdc_0')
    print_matrix(sess.run(end_points['output_sdc_1']), 'output_sdc_1')
    print_matrix(sess.run(tf.get_default_graph().get_tensor_by_name('Model/conv1/weights:0')), 'weights_output_sdc_0')
    print_matrix(sess.run(tf.get_default_graph().get_tensor_by_name('Model/conv1/biases:0')), 'biases_output_sdc_0')
    print_matrix(sess.run(tf.get_default_graph().get_tensor_by_name('Model/input_recovery/weights:0')),
                 'weights_input_sdc_1')
    print_matrix(sess.run(tf.get_default_graph().get_tensor_by_name('Model/input_recovery/biases:0')),
                 'biases_input_sdc_1')


def assign_weight_biases():
    weights_data = np.array(
        [[
            [-0.5, 0.4],
            [0.6, 0.5],
        ], [
            [0.1, -0.2],
            [0.7, 0.1],
        ]])
    biases_output = np.array([0.005, 0.002])

    weights = np.zeros(shape=(2, 2, 1, 2))
    weights[0][0][0][0] = weights_data[0][0][0]
    weights[0][1][0][0] = weights_data[0][0][1]
    weights[1][0][0][0] = weights_data[0][1][0]
    weights[1][1][0][0] = weights_data[0][1][1]
    weights[0][0][0][1] = weights_data[1][0][0]
    weights[0][1][0][1] = weights_data[1][0][1]
    weights[1][0][0][1] = weights_data[1][1][0]
    weights[1][1][0][1] = weights_data[1][1][1]

    biases_input = np.array([0.003])

    with tf.variable_scope('Model', reuse=True):
        return [tf.get_variable('conv1/weights').assign(weights),
                tf.get_variable('conv1/biases').assign(biases_output),
                tf.get_variable('input_recovery/weights').assign(weights),
                tf.get_variable('input_recovery/biases').assign(biases_input)]


# input_sdc_0 = np.matrix([[0.5, 0.6],
#                          [0.4, 0.7]])
input_sdc_0 = np.random.rand(2352)

input = ops.convert_to_tensor(input_sdc_0, dtype=tf.float32, name='input_sdc_0')
input = tf.reshape(input, shape=(1, 28, 28, 3))

model_ae, end_points = model(input)

loss_input = tf.reduce_sum(tf.divide(tf.square(end_points['input_sdc_1'] - end_points['input_sdc_0']),
                                     tf.constant(2.0)))
loss_output = tf.reduce_sum(tf.divide(tf.square(end_points['output_sdc_1'] - end_points['output_sdc_0']),
                                      tf.constant(2.0)))
loss = loss_input + loss_output
tf.losses.add_loss(loss)

# Create custom gradient for SDC1
grad = {}
for var in tf.trainable_variables():
    grad[var.name] = tf.placeholder(tf.float32, shape=var.shape, name=var._shared_name + '_grad')

optimizer = sgd_custom(grad=grad, learning_rate=alpha)

train_op = slim.learning.create_train_op(loss, optimizer)

# list_assigned = assign_weight_biases()

################################
# SUMMARIES #
################################

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

summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES))

# Merge all summaries together.
summary_op = tf.summary.merge(list(summaries), name='summary_op')

###########################################################################################
# USER PARAMS for custom gradient
###########################################################################################
activation_name = str.lower(end_points['output_sdc_0'].name.split('/')[-1].split(':')[0])
if activation_name == 'maximum':
    activation_name = str.lower(end_points['output_sdc_0'].name.split('/')[2])

################################
logdir = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/simple_sdc_1/'

sv = tf.train.Supervisor(logdir=logdir,
                         graph=tf.get_default_graph(),
                         summary_op=summary_op,
                         global_step=0)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

with sv.managed_session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    # sess.run(list_assigned)
    # log(sess)

    timer_list = []
    for i in range(max_step):

        # Calc custom gradient
        start = timer()
        x = sess.run([end_points['input_sdc_0'], end_points['input_sdc_1']])
        y = sess.run([end_points['output_sdc_0'], end_points['output_sdc_1']])
        timer_list.append(timer() - start)

        # Create custom gradient for autoencoder_sdc_1

        start = timer()
        grad_custom = gradient_custom(grads=grad,
                                      x=x,
                                      y=y,
                                      stride=stride,
                                      padding=padding,
                                      formulas=formulas,
                                      activation_name=activation_name)
        grad_values = grad_custom.run()
        timer_list.append(timer() - start)

        # Train autoencoder
        start = timer()
        cost = sess.run(train_op, feed_dict=grad_values)
        timer_list.append(timer() - start)

        if (i + 1) % log_step == 0:
            print(str(i + 1) + ') Loss: {:.5f}; "x" and "y": {:.3f}; grad_custom: {:.6f}; sess.run: {:.3f}'.format(cost,
                                                                                                                   timer_list[
                                                                                                                       0],
                                                                                                                   timer_list[
                                                                                                                       1],
                                                                                                                   timer_list[
                                                                                                                       2]))
        timer_list.clear()

        if (i + 1) % max_step + 1 == 0:
            log(sess)

    checkpoint_path = logdir + 'model.ckpt'
    sv.saver.save(sess, save_path=checkpoint_path, global_step=max_step)
    sv.saver.export_meta_graph(logdir + 'graph.pbtxt')
