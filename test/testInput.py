# coding=utf-8
'''
Created on 2019年1月12日

@author: zhang
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
FLAGS = tf.flags.FLAGS
print(tf.__version__)


def parse_fn(example):
    "Parse TFExample records and perform simple data augmentation."
    example_fmt = {
        "label": tf.FixedLenFeature([1], tf.int64),
        "embedding": tf.FixedLenFeature([2], tf.float32)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    return parsed


def input_fn():
    files = tf.data.Dataset.list_files(
        "hdfs://namenodefd1v.qss.zzzc.qihoo.net:9000/home/hdp-reader/proj/hdp-reader-vqt/zhangyanqing1/tfrecord/*.tf")
    #ds = files.interleave(tf.data.TFRecordDataset, cycle_length=1)
    ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=FLAGS.num_parallel_readers))

    ds = ds.shuffle(buffer_size=FLAGS.batch_size * 4)

    ds = ds.apply(tf.contrib.data.map_and_batch(
        map_func=parse_fn, batch_size=FLAGS.batch_size))

    ds = ds.prefetch(
        buffer_size=FLAGS.prefetch_buffer_size)  # last transformation
    return ds


def input_fn_local():
    files = tf.data.Dataset.list_files("../data/*.tf", shuffle=False)

    #ds = files.interleave(tf.data.TFRecordDataset, cycle_length=1)
    ds = files.apply(tf.contrib.data.parallel_interleave(
        tf.data.TFRecordDataset, cycle_length=4))

    ds = ds.shuffle(buffer_size=2 * 4)

    ds = ds.apply(tf.contrib.data.map_and_batch(
        map_func=parse_fn, batch_size=5))

    ds = ds.repeat(1000)

    ds = ds.prefetch(
        buffer_size=2)  # last transformation

    return ds


def build_input_fn():
    return input_fn_local


def shard_fn():
    files = tf.data.Dataset.list_files('./*', shuffle=False)
    files0 = files.shard(4, 0)
    files1 = files.shard(4, 1)
    files2 = files.shard(4, 2)
    files3 = files.shard(4, 3)
    return files0, files1, files2, files3


if __name__ == '__main__':
    ds = input_fn_local()
    ds = ds.make_one_shot_iterator()
    ds = ds.get_next()

    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(ds))

            except tf.errors.OutOfRangeError:
                break
