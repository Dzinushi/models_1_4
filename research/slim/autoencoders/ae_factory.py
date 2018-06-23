from tensorflow.contrib import slim
from research.slim.autoencoders.lenet import lenet_bm
import functools

ae_map = {'lenet_bm': lenet_bm.lenet_bm}

ae_main_ls = {'lenet_bm': lenet_bm.get_main_ls}


def get_ae_fn(name):
    if name not in ae_map[name]:
        raise ValueError('Name of network unknown %s' % name)
    func_ae = ae_map[name]
    func_ae_main_ls = ae_main_ls[name]

    @functools.wraps(func_ae)
    def autoencoder_fn(images, **kwards):
        return func_ae(images, **kwards)

    @functools.wraps(func_ae_main_ls)
    def autoencoder_main_layer_scope():
        return func_ae_main_ls()

    if hasattr(func_ae, 'default_image_size'):
        autoencoder_fn.default_image_size = func_ae.default_image_size

    return autoencoder_fn, autoencoder_main_layer_scope
