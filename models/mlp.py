# -*- coding: utf-8 -*-
# vim: set ts=2 sw=2 sts=2 tw=80 et:
# pylint: disable=missing-docstring

import tensorflow as tf


def mlp(inputs, mode="train", batch_norm=True, dropout=True, weight_decay=0.0,
        layer_sizes=None, activations=None, trainables=None, model_prefix=''):
    """
    """
    layer_sizes = layer_sizes if layer_sizes is not None else []
    activations = activations if activations is not None else []
    trainables = trainables if trainables is not None else []

    is_training = bool(mode == 'train')
    he_initializer = tf.contrib.layers.variance_scaling_initializer(
        factor=2.0, mode='FAN_AVG', uniform=True)
    weights_regularizer = None
    if weight_decay != 0:
        weights_regularizer = tf.contrib.layers.l2_regularizer(
            weight_decay)

    this_layer = inputs

    for idx, next_layer_size in enumerate(layer_sizes):
        this_layer_trainable = trainables[idx]

        with tf.variable_scope("%s_mlp_%d" % (model_prefix, idx), reuse=tf.AUTO_REUSE) as scope:
            this_layer = tf.contrib.layers.fully_connected(this_layer, next_layer_size,
                                                           activation_fn=None,
                                                           trainable=this_layer_trainable,
                                                           weights_initializer=he_initializer,
                                                           weights_regularizer=weights_regularizer,
                                                           scope=scope)
            if batch_norm:
                this_layer = tf.contrib.layers.batch_norm(this_layer,
                                                          center=True,
                                                          scale=True,
                                                          is_training=is_training,
                                                          trainable=this_layer_trainable,
                                                          scope=scope)

            this_layer = activations[idx](this_layer)

            if dropout:
                if is_training:
                    this_layer = tf.nn.dropout(this_layer, 0.9, name='dropout')
                else:
                    pass

    return this_layer


def network(inputs, embedding_size, activation="relu", mode="train",
            enable_dropout=True, weight_decay=0.0):
    """
    Args:
      embedding_size: inner embedding dim size
    """
    with tf.variable_scope("title"):
        layer_sizes = [8, 6, embedding_size]
        trainables = [True, True, True]

        if activation == "relu":
            activations = [tf.nn.relu, tf.nn.relu, tf.nn.relu]
        elif activation == "tanh":
            activations = [tf.nn.tanh, tf.nn.tanh, tf.nn.tanh]
        else:
            raise ValueError('unknown activation: %s' % activation)

        output = mlp(inputs, mode=mode,
                     batch_norm=True,
                     dropout=enable_dropout,
                     weight_decay=weight_decay,
                     layer_sizes=layer_sizes,
                     activations=activations,
                     trainables=trainables)

    return output
