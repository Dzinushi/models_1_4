import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
from research.slim.autoencoders.optimizers.sgd_sdc_1 import GradientDescentOptimizerSDC1
from tensorflow.contrib import slim
from collections import defaultdict
from research.slim.autoencoders.optimizers.gradient_sdc_1 import GradientSDC1, Formulas, DActivation


def model(input):
    with tf.variable_scope('Model'):
        end_point = {}
        end_point['input_sdc_0'] = net = input
        end_point['output_sdc_0'] = net = slim.conv2d(net, 2, [2, 2], padding='VALID', activation_fn=tf.nn.tanh,
                                                      scope='conv1')
        end_point['input_sdc_1'] = net = slim.conv2d_transpose(net, 1, [2, 2],
                                                               padding='VALID', scope='input_recovery')
        end_point['output_sdc_1'] = net = slim.conv2d(net, 2, [2, 2], reuse=True, padding='VALID',
                                                      activation_fn=tf.nn.tanh,
                                                      scope='conv1')
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


input_sdc_0 = np.matrix([[0.5, 0.6],
                         [0.4, 0.7]])

input = ops.convert_to_tensor(input_sdc_0, dtype=tf.float32, name='input_sdc_0')
input = tf.reshape(input, shape=(1, 2, 2, 1))

model, end_points = model(input)

# input_sdc_1_t = ops.convert_to_tensor(input_sdc_1, dtype=tf.float32, name='input_sdc_1')
# output_sdc_0_t = ops.convert_to_tensor(output_sdc_0, dtype=tf.float32, name='output_sdc_0')
# output_sdc_1_t = ops.convert_to_tensor(output_sdc_1, dtype=tf.float32, name='output_sdc_1')

loss_input = tf.reduce_sum(tf.divide(tf.square(end_points['input_sdc_1'] - end_points['input_sdc_0']),
                                     tf.constant(2.0)))
loss_output = tf.reduce_sum(tf.divide(tf.square(end_points['output_sdc_1'] - end_points['output_sdc_0']),
                                      tf.constant(2.0)))
loss = loss_input + loss_output
tf.losses.add_loss(loss)

# Create custom gradient for SDC1
# grad = tf.placeholder(tf.float32, shape=weight_shape, name=)
grad = {}
for var in tf.trainable_variables():
    grad[var.name] = tf.placeholder(tf.float32, shape=var.shape, name=var._shared_name + '_grad')
optimizer = GradientDescentOptimizerSDC1(grad=grad, learning_rate=0.01)
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# params = tf.trainable_variables()
# gradients = tf.gradients(loss, params)
# optimizer.apply_gradients(zip(gradients, params))

gradient = optimizer.compute_gradients(loss)

train_op = slim.learning.create_train_op(loss, optimizer)

list_assigned = assign_weight_biases()

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

################################
logdir = '/media/w_programs/Development/Python/tf_autoencoders/checkpoints/simple_sdc_1/'

step = 1000

sv = tf.train.Supervisor(logdir=logdir,
                         graph=tf.get_default_graph(),
                         summary_op=summary_op,
                         global_step=0)

with sv.managed_session() as sess:
    sess.run(list_assigned)
    # log(sess)

    for i in range(step):
        # Calc custom gradient
        x = sess.run([end_points['input_sdc_0'], end_points['input_sdc_1']])
        y = sess.run([end_points['output_sdc_0'], end_points['output_sdc_1']])

        # Create custom gradient for autoencoder_sdc_1
        grad_custom = GradientSDC1(grads=grad,
                                   x=x,
                                   y=y,
                                   stride=1,
                                   padding='VALID',
                                   formulas=Formulas.golovko,
                                   d_activation_name=DActivation.sigmoid)

        # Train autoencoder
        cost = sess.run(train_op, feed_dict=grad_custom.calc())

        print(str(i + 1) + ') Loss: ', cost)
        if (i + 1) % 1001 == 0:
            log(sess)

    checkpoint_path = logdir + 'model.ckpt'
    sv.saver.save(sess, save_path=checkpoint_path, global_step=step)
    sv.saver.export_meta_graph(logdir + 'graph.pbtxt')


