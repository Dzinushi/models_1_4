import numpy as np

input_sdc_0 = np.array([
    [0.5, 0.6000000238418579],
    [0.4000000059604645, 0.699999988079071],
])
input_sdc_1 = np.array([
    [
        0.0,
        0.6360000371932983
    ],
    [
        0.0,
        0.5037000179290771
    ],
])
output_sdc_0 = np.array([
    0.6450000405311584,
    0.4919999837875366
])
output_sdc_1 = np.array([
    0.7391900420188904,
    0.3703700304031372
])
weights_output_sdc_0 = np.array([
    [
        [
            [-0.5, 0.4]
        ],
        [
            [0.6, 0.5]
        ]
    ],
    [
        [
            [0.1, -0.2]
        ],
        [
            [0.7, 0.1]
        ]
    ]
])
biases_output_sdc_0 = np.array([
    0.004999999888241291,
    0.0020000000949949026
])
weights_input_sdc_1 = np.array([
    [
        [
            -0.5,
            0.4000000059604645
        ],
        [
            0.6000000238418579,
            0.5
        ],
    ],
    [
        [
            0.10000000149011612,
            -0.20000000298023224
        ],
        [
            0.699999988079071,
            0.10000000149011612
        ],
    ],
])
biases_input_sdc_1 = np.array([
    0.003000000026077032
])


############################################################
# OPERATIONS
############################################################
def grad_weights(input_sdc_0, input_sdc_1, output_sdc_0, output_sdc_1):
    assert input_sdc_0.shape == input_sdc_1.shape
    assert output_sdc_0.shape == output_sdc_1.shape

    w_shape = weights_output_sdc_0.shape
    grad = np.zeros(shape=w_shape)

    for width in range(w_shape[0]):
        for height in range(w_shape[1]):
            for maps in range(w_shape[2]):
                grad[width][height][maps] = (output_sdc_1[width][height][maps] - output_sdc_0[width][height][maps]) * \
                                            input_sdc_1[width][
                                                height] \
                                            + (input_sdc_1[width][height] - input_sdc_0[width][height]) * \
                                            output_sdc_0[width][height][maps]
    print(grad)
    return grad


def grad_biases(layer_sdc_0, layer_sdc_1):
    assert layer_sdc_0.shape == layer_sdc_1.shape

    b_shape = layer_sdc_0.shape
    grad = np.zeros(shape=b_shape)

    if len(b_shape) == 3:
        for width in range(b_shape[0]):
            for height in range(b_shape[1]):
                for maps in range(b_shape[2]):
                    grad[width][height][maps] = layer_sdc_1[width][height][maps] - layer_sdc_0[width][height][maps]
    elif len(b_shape) == 2:
        for width in range(b_shape[0]):
            for height in range(b_shape[1]):
                grad[width][height] = layer_sdc_1[width][height] - layer_sdc_0[width][height]
    else:
        ValueError('Gradient shape broken: ', len(b_shape))
    print(grad)
    return grad


weights_output_sdc_0 = weights_output_sdc_0 - 0.01 * grad_weights(input_sdc_0, input_sdc_1, output_sdc_0, output_sdc_1)
biases_output_sdc_0 = biases_output_sdc_0 - 0.01 * grad_biases(output_sdc_1, output_sdc_0)
biases_input_sdc_1 = biases_input_sdc_1 - 0.01 * grad_biases(input_sdc_1, input_sdc_0)

loss_input = np.sum(np.divide(np.square(input_sdc_1 - input_sdc_0),
                              2.0))
loss_output = np.sum(np.divide(np.square(output_sdc_1 - output_sdc_0),
                               2.0))
loss = loss_input + loss_output
print('Loss: ', loss)
