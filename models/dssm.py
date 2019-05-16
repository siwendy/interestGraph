import tensorflow as tf
import numpy as np
import time

def mlp(x, config):
    sizes = config['mlp_sizes']
    activation = config['activation']
    bn = config.get('bn', False)
    is_training = config.get('is_training', False)
    wd = config.get('wd', 0.0)
    shape = x.get_shape()
    assert shape.ndims == 2
    tmp = x
    for idx, size in enumerate(sizes):
        prev_size = tmp.get_shape()[1].value
        with tf.variable_scope("linear_%d" % idx) as scope:
            weight = tf.get_variable(
                'weight',
                shape=[prev_size, size],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=(tf.contrib.layers.l2_regularizer(wd) if wd != 0 else None))
            bias = tf.get_variable('bias', initializer=tf.zeros([size]))
            tmp = tf.matmul(tmp, weight) + bias
            if not tf.get_variable_scope().name.startswith('consolidated') and \
                    bn and activation[idx] != tf.identity:
                tmp = tf.contrib.layers.batch_norm(
                    tmp, scale=True, is_training=is_training,
                    variables_collections={'moving_mean': ["MOVING"], 'moving_variance': ["MOVING"]},
                    updates_collections=None)

                scope.reuse_variables()
                moving_mean = tf.get_variable('BatchNorm/moving_mean')
                moving_variance = tf.get_variable('BatchNorm/moving_variance')
                gamma = tf.get_variable('BatchNorm/gamma')
                beta = tf.get_variable('BatchNorm/beta')
                tf.add_to_collection('consolidate', [moving_mean, moving_variance, beta, gamma, weight, bias])
        tmp = activation[idx](tmp)
    return tmp

def network(x, counts, **kwargs):
    config = {
        'embed_activation': tf.nn.relu,
        'activation': [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu],
        'mlp_sizes': [300, 256, 256, 128],
        'bn': True,
    }
    config.update(kwargs)
    x_shape = x.get_shape() # [batch_size, num_per_entry, max_sentence_len, embed_dim]
    num_per_entry = tf.shape(x)[1]
    x = tf.reshape(x, tf.stack([-1, x_shape[2].value, x_shape[3].value]))
    x = tf.reduce_sum(x, 1)
    x = config['embed_activation'](x)
    x = mlp(x, config)
    x = tf.reshape(x, tf.stack([-1, num_per_entry, config['mlp_sizes'][-1]]))
    return x
