# -*- coding: utf-8 -*-
# vim: set ts=2 sw=2 sts=2 tw=80 et:
# pylint: disable=missing-docstring

import tensorflow as tf
import numpy as np


def mlp(inputs, mode="train", batch_norm=True, dropout=True, weight_decay=0.0,
        layer_sizes=None, activations=None, trainables=None, **kwargs):
    print(mode)
    """
  """
    layer_sizes = layer_sizes if layer_sizes is not None else []
    activations = activations if activations is not None else []
    trainables = trainables if trainables is not None else []

    this_layer = inputs

    for idx, layer_size in enumerate(layer_sizes):
        prev_layer_size = this_layer.get_shape()[1].value

        this_layer_trainable = trainables[idx]

        with tf.variable_scope("linear_%d" % idx, reuse=tf.AUTO_REUSE) as scope:
            weight = tf.get_variable(
                'weight',
                shape=[prev_layer_size, layer_size],
                trainable=this_layer_trainable,
                initializer=tf.contrib.layers.xavier_initializer(),
                regularizer=(tf.contrib.layers.l2_regularizer(
                    weight_decay) if weight_decay != 0 else None)
            )

            bias = tf.get_variable(
                'bias',
                trainable=this_layer_trainable,
                initializer=tf.zeros([layer_size])
            )

            #this_layer = tf.matmul(this_layer, weight) + bias
            this_layer = tf.nn.xw_plus_b(
                this_layer, weight, bias, name='xw_plus_b')

            if batch_norm:
                this_layer = tf.contrib.layers.batch_norm(
                    this_layer,
                    scale=True,
                    is_training=bool(mode == 'train'),
                    trainable=this_layer_trainable,
                    updates_collections=None
                )

            this_layer = activations[idx](this_layer)

            if dropout:
                if mode == "train":
                    this_layer = tf.nn.dropout(this_layer, 0.9, name='dropout')
                else:
                    pass

            #tf.summary.histogram('layer_%d_output' % idx, this_layer)

    return this_layer


def network(inputs, embedding_size, activation="relu", mode="train",
            enable_dropout=True, weight_decay=0.0, **kwargs):
    """
    Args:
      embedding_size: inner embedding dim size
    """
    with tf.variable_scope("title") as scope:
        layer_sizes = [8192, 1024, embedding_size]
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
