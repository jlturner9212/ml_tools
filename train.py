import tensorflow as tf

__weights_dict = dict()

is_train = False

def load_weights(weight_file):
    import numpy as np

    if weight_file == None:
        return

    try:
        weights_dict = np.load(weight_file).item()
    except:
        weights_dict = np.load(weight_file, encoding='bytes').item()

    return weights_dict


def KitModel(weight_file = None):
    global __weights_dict
    __weights_dict = load_weights(weight_file)

    Input_1         = tf.placeholder(tf.float32,  shape = (None, 1, 28, 28), name = 'Input_1')
    Convolution2D_1 = convolution(Input_1, group=1, strides=[1, 1], padding='VALID', name='Convolution2D_1')
    Convolution2D_1_activation = tf.nn.relu(Convolution2D_1, name = 'Convolution2D_1_activation')
    Convolution2D_2 = convolution(Convolution2D_1_activation, group=1, strides=[1, 1], padding='VALID', name='Convolution2D_2')
    Convolution2D_2_activation = tf.nn.relu(Convolution2D_2, name = 'Convolution2D_2_activation')
    MaxPooling2D_1  = tf.nn.max_pool(Convolution2D_2_activation, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID', name='MaxPooling2D_1')
    Flatten_1       = tf.contrib.layers.flatten(MaxPooling2D_1)
    Dense_1         = tf.layers.dense(Flatten_1, 128, kernel_initializer = tf.constant_initializer(__weights_dict['Dense_1']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['Dense_1']['bias']), use_bias = True)
    Dense_1_activation = tf.nn.relu(Dense_1, name = 'Dense_1_activation')
    Dense_2         = tf.layers.dense(Dense_1_activation, 10, kernel_initializer = tf.constant_initializer(__weights_dict['Dense_2']['weights']), bias_initializer = tf.constant_initializer(__weights_dict['Dense_2']['bias']), use_bias = True)
    Dense_2_activation = tf.nn.softmax(Dense_2, name = 'Dense_2_activation')
    return Input_1, Dense_2_activation


def convolution(input, name, group, **kwargs):
    w = tf.Variable(__weights_dict[name]['weights'], trainable=is_train, name=name + "_weight")
    if group == 1:
        layer = tf.nn.convolution(input, w, **kwargs)
    else:
        weight_groups = tf.split(w, num_or_size_splits=group, axis=-1)
        xs = tf.split(input, num_or_size_splits=group, axis=-1)
        convolved = [tf.nn.convolution(x, weight, **kwargs) for
                    (x, weight) in zip(xs, weight_groups)]
        layer = tf.concat(convolved, axis=-1)

    if 'bias' in __weights_dict[name]:
        b = tf.Variable(__weights_dict[name]['bias'], trainable=is_train, name=name + "_bias")
        layer = layer + b
    return layer
