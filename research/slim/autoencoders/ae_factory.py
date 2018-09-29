from autoencoders.nets_bm.lenet import lenet_bm
from autoencoders.nets_bm.mobilenet_v1 import mobilenet_v1_bm
from autoencoders.nets_bm.alexnet import alexnet_bm
import functools

ae_map = {'lenet_bm': lenet_bm.lenet_bm,
          'alexnet_bm': alexnet_bm.alexnet_v2
          }

ae_loss_map = {'lenet_bm': lenet_bm.lenet_model_losses,
               'alexnet_bm': alexnet_bm.alexnet_model_losses
               }


def get_ae_fn(name):
    if name not in ae_map:
        raise ValueError('Name of network unknown %s' % name)
    func_ae = ae_map[name]
    func_ae_loss_layer_names = ae_loss_map[name]

    @functools.wraps(func_ae)
    def autoencoder_fn(images, **kwards):
        return func_ae(images, **kwards)

    @functools.wraps(func_ae_loss_layer_names)
    def autoencoder_loss_map_fn(end_points, **kwards):
        return func_ae_loss_layer_names(end_points, **kwards)

    if hasattr(func_ae, 'default_image_size'):
        autoencoder_fn.default_image_size = func_ae.default_image_size

    return autoencoder_fn, autoencoder_loss_map_fn
