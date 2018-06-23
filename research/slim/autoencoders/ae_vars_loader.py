import tensorflow as tf
from research.slim.autoencoders.utils import load_layer_vars


def rename_optimizer(optimizer_name):
    if optimizer_name == 'rmsprop':
        return 'RMSProp'
    elif optimizer_name == 'adam':
        return 'Adam'
    else:
        raise ValueError('Autoencoder not recognized --optimizer')


def load_assign_vars(ae_model_path, optimizer_name, ae_main_layer_scope, graph, session):
    load_sess = tf.Session()
    with load_sess:
        # Load graph
        saver = tf.train.import_meta_graph(ae_model_path + '.meta')
        saver.restore(load_sess, tf.train.latest_checkpoint('/'.join(ae_model_path.split('/')[:-1])))
        load_graph = tf.get_default_graph()

        # Rename optimizer
        optimizer_name = rename_optimizer(optimizer_name)

        assign_layer = lambda layer_scope: load_layer_vars(
            l_graph=load_graph,
            l_session=load_sess,
            graph=graph,
            session=session,
            scope=scope,
            layer_scope=layer_scope)

        assign_layer('conv1')
        assign_layer('conv2')
        assign_layer('fc3')
