# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import math

import base64
from random import randint
from random import shuffle
import collections

import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
logger = tf.logging
flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("input_filename", "interest", "interest file name")
flags.DEFINE_string("distribution_filename",
                    "interest_distribution", "interest distribution")
flags.DEFINE_string("tfRecord_filename", "tfRecord",
                    "output tfRecord filename")
flags.DEFINE_integer("title_embedding_fld", 6, "embedding field")
flags.DEFINE_integer("num_negative", 0, "negative sample number")
flags.DEFINE_integer("label_fld", 0, "label field")
flags.DEFINE_boolean("enable_resample", False, "enable resample")
flags.DEFINE_boolean("enable_shuffle", False, "enable shuffle")

counter = collections.Counter()


def main(_):
    interest_id_list = []
    interest_p_list = []

    with open(FLAGS.distribution_filename) as infile:
        for line in infile:
            flds = line.strip().split("\t")
            if len(flds) != 2:
                raise ValueError('distribution file fields error')
            interest_id_list.append(int(flds[0]))
            interest_p_list.append(float(flds[1]))

    interest_p_list = np.array(interest_p_list)
    interest_p_list /= interest_p_list.sum()  # normalize
    max_interest_p = max(interest_p_list)
    interest_p_map = dict(zip(interest_id_list, interest_p_list))

    writer = tf.python_io.TFRecordWriter(FLAGS.tfRecord_filename)
    example_list = []
    with open(FLAGS.input_filename, 'r') as f:
        for line in f:
            flds = line.strip().split("\t")
            label_id = int(flds[FLAGS.label_fld])

            if FLAGS.enable_resample:
                interest_p = interest_p_map[label_id]
                weight = int(math.sqrt(max_interest_p / interest_p))
            else:
                weight = 1
            counter.update(["Weight %d" % weight])
            try:
                title_embedding = np.frombuffer(base64.b64decode(
                    flds[FLAGS.title_embedding_fld]), dtype=np.float32).reshape((-1))
            except:
                logger.error("base64 error\t" + line)
                counter.update(["base64 error"])
                continue

            for _ in range(weight):
                features = np.append(title_embedding, label_id)
                for _ in range(FLAGS.num_negative):
                    while True:
                        negative_cid = randint(1, len(interest_p_map))
                        if negative_cid != label_id:
                            break
                    features = np.append(features, negative_cid)

                example = tf.train.Example(features=tf.train.Features(feature={
                    "f": tf.train.Feature(float_list=tf.train.FloatList(value=features))
                }))

                example_list.append(example)

                if len(example_list) >= 100000:
                    if FLAGS.enable_shuffle:
                        shuffle(example_list)
                    for example in example_list:
                        sys.stderr.write("reporter:counter:Stat,Sample,1\n")
                        writer.write(example.SerializeToString())
                    del example_list[:]

    if FLAGS.enable_shuffle:
        shuffle(example_list)
    for example in example_list:
        sys.stderr.write("reporter:counter:Stat,Sample,1\n")
        writer.write(example.SerializeToString())


if __name__ == '__main__':
    flags.mark_flag_as_required("distribution_filename")
    flags.mark_flag_as_required("title_embedding_fld")
    flags.mark_flag_as_required("label_fld")
    tf.app.run()
