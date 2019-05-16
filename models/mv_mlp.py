import tensorflow as tf
import numpy as np
import time


def mlp(x, config):
    sizes = config['mlp_sizes']
    activation = config['activation']
    bn = config.get('bn')
    mode = config.get('mode')
    enable_dropout = config.get('enable_dropout')
    wd = config.get('wd')
    shape = x.get_shape()
    assert shape.ndims == 2
    tmp = x
    for idx, size in enumerate(sizes):
        prev_size = tmp.get_shape()[1].value
        with tf.variable_scope("linear_%d" % idx, reuse=tf.AUTO_REUSE) as scope:
            weight = tf.get_variable(
                'weight',
                shape=[prev_size, size],
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=(tf.contrib.layers.l2_regularizer(wd) if wd != 0 else None))
            bias = tf.get_variable('bias', initializer=tf.zeros([size]))
            tmp = tf.matmul(tmp, weight) + bias
            if bn and activation[idx] != tf.identity:
                if mode == "train":
                    tmp = tf.contrib.layers.batch_norm(
                        tmp, scale=True, is_training=True,
                        updates_collections=None)
                else:
                    tmp = tf.contrib.layers.batch_norm(
                        tmp, scale=True, is_training=False,
                        updates_collections=None)
        tmp = activation[idx](tmp)
        if enable_dropout and mode == "train":
            tmp = tf.nn.dropout(tmp, 0.9)
    return tmp


def network(data, num_cluster, activation="relu", title_embedding_size=512, embedding_size=512, mode="train", enable_dropout=False, weight_decay=0.0, max_token_num=30):
    tensor_1 = tf.slice(data, [0, 0], [-1, title_embedding_size])
    tensor_2 = tf.slice(data, [0, title_embedding_size], [-1, num_cluster])
    # tensor_1 = tf.slice(data, [0, 0], [-1, max_token_num*title_embedding_size])
    # tensor_2 = tf.slice(data, [0, max_token_num*title_embedding_size], [-1, num_cluster])
    with tf.variable_scope("title", reuse=tf.AUTO_REUSE) as scope:
        if activation == "relu":
            embed_activation = tf.nn.relu,
            activation = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
        else:
            embed_activation = tf.nn.tanh,
            activation = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.tanh]
        config_1 = {
            'embed_activation': embed_activation,
            'activation': activation,
            'mlp_sizes': [8192, 1024, 512],
            'bn': True,
        }
        config_1.update(kwargs)
        tensor_1 = mlp(tensor_1, config_1)
        tensor_1 = tf.reshape(tensor_1, [-1, 1, config_1['mlp_sizes'][-1]])

    with tf.variable_scope("cluster", reuse=tf.AUTO_REUSE) as scope:
        tensor_2 = tf.cast(tensor_2, tf.int32)
        cluster_embedding_size = 512
        tensor_2 = tf.reshape(tensor_2, [-1, num_cluster, 1])
        emb_param = tf.get_variable(
            'emb_param', shape=[cluster_num + 1, cluster_embedding_size],
            initializer=tf.random_uniform_initializer(-0.1, 0.1),
            regularizer=None)
        tensor_2 = tf.nn.embedding_lookup(emb_param, tensor_2)
        if activation == "relu":
            embed_activation = tf.nn.relu,
            activation = [tf.nn.relu, tf.nn.relu, tf.nn.relu, tf.nn.relu]
        else:
            embed_activation = tf.nn.tanh,
            activation = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh, tf.nn.tanh]
        config_2 = {
            'embed_activation': embed_activation,
            'activation': activation,
            'mlp_sizes': [512, 512],
            'bn': True,
        }
        config_2.update(kwargs)
        num_per_entry = num_cluster
        tensor_2 = tf.reshape(tensor_2, [-1, cluster_embedding_size])
        # tensor_2 = tf.reduce_sum(tensor_2, 1)
        # tensor_2 = config_2['embed_activation'](tensor_2)
        tensor_2 = mlp(tensor_2, config_2)
        # [batch_size, 1, output_size]
        tensor_2 = tf.reshape(tensor_2,
                              [-1, num_per_entry, config_2['mlp_sizes'][-1]])

    return tf.concat([tensor_1, tensor_2], 1)
