'''
Created on 2018年12月15日

@author: zhang
'''
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf

with tf.variable_scope("scope0") as scope:
    tmp = tf.constant([[4., 5., 6.]])
    weight = tf.get_variable(
        'weight',
        shape=[3, 4],
        trainable=False,
        initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('bias', initializer=tf.zeros([4]))
    tmp = tf.matmul(tmp, weight) + bias

with tf.variable_scope("scope1") as scope:
    tmp = tf.constant([[4., 5., 6.]])
    weight = tf.get_variable(
        'weight',
        shape=[3, 4],
        initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('bias', initializer=tf.zeros([4]))
    tmp = tf.matmul(tmp, weight) + bias

vgg_ref_vars = tf.get_collection(
    tf.GraphKeys.TRAINABLE_VARIABLES, scope='scope0')
saver = tf.train.Saver(vgg_ref_vars)

init_ops = [tf.global_variables_initializer(),
            tf.local_variables_initializer()]


config = tf.ConfigProto(allow_soft_placement=True,
                        log_device_placement=False)
#    intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
config.gpu_options.allow_growth = True
#sess = tf.Session(config=config)
with tf.Session(config=config) as sess:
    sess.run(init_ops)

    w, b, a = sess.run([weight, bias, tmp])
    print(w)
    print(b)
    print(a)
    saver.save(sess, 'D:/ckpt')
