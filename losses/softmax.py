# -*- coding: utf-8 -*-
# vim: set ts=2 sw=2 sts=2 tw=80 et:
# pylint: disable=missing-docstring

import tensorflow as tf


def loss(inputs, labels, num_classes, mode="train", trainable=True, weight_decay=0.0, **kwargs):
    """Softmax loss

    Args:
      inputs:
      labels:
      num_classes
      mode
      trainable
      weight_decay

    Returns:
      Tuple of:
        loss:
        correct
    """
    with tf.variable_scope("linear_softmax", reuse=tf.AUTO_REUSE) as scope:
        logits = tf.contrib.layers.fully_connected(inputs, num_classes,
                                                   activation_fn=None,
                                                   trainable=trainable,
                                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                   weights_regularizer=tf.contrib.layers.l2_regularizer(weight_decay) if
                                                   bool(weight_decay) else None,
                                                   biases_initializer=tf.zeros_initializer(),
                                                   scope=scope)

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=tf.reshape(labels, [-1]))
        loss = tf.reduce_mean(loss)

        if mode == "train":
            tf.summary.scalar('train_loss', loss)
            return loss
        else:
            predictions = tf.nn.softmax(logits)
            value, indices = tf.nn.top_k(predictions, 2)

            in_top_k = tf.nn.in_top_k(predictions, tf.reshape(labels, [-1]), 1)
            correct = tf.reduce_sum(tf.cast(in_top_k, tf.int32))

            if mode == "eval":
                return loss, correct, indices, labels
            elif mode == "test":
                return indices, value, correct
            elif mode == 'debug':
                return loss, correct, indices, labels, logits, inputs
