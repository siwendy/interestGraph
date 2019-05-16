'''
Created on 2018年12月15日

@author: zhang
'''
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import tensorflow as tf
from utils.tfRecordManager import TFRecordConvertor
# tf.enable_eager_execution()
tf.logging.set_verbosity(tf.logging.INFO)

featureNames = ["name", "value", "fvalue"]
featureTypes = [tf.string, tf.int64, tf.float32]
featureDims = [1, 3, 2]
tm = TFRecordConvertor(featureNames, featureTypes, featureDims)


#file_names = tf.placeholder(tf.string)
source_files = tf.train.match_filenames_once('../data/*.txt')
source_data = tf.data.TextLineDataset(source_files).repeat(1).shuffle(1)
source_iterator = source_data.make_initializable_iterator()
line = source_iterator.get_next()

#tf_files = tf.train.match_filenames_once('*.tf')
tf_files = tf.data.TFRecordDataset.list_files('test.tf', shuffle=False)
tf_data = tf.data.TFRecordDataset(tf_files,
                                  compression_type=None,
                                  buffer_size=8 * 1024,
                                  num_parallel_reads=1
                                  ).map(tm.bytesToTensor).skip(0).shuffle(1).repeat(1)
#.batch(1, True)

tf_iterator = tf_data.make_initializable_iterator()

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(
    handle, tf_data.output_types, tf_data.output_shapes)
next_element = iterator.get_next()


init = (tf.global_variables_initializer(), tf.local_variables_initializer())
with tf.Session() as sess:
    sess.run(init)
    sess.run(source_iterator.initializer)

    writer = tf.python_io.TFRecordWriter("testSet.tf")
    while True:
        try:
            s = sess.run(line)
            s = s.decode()

            flds = s.split('\t')
            name = [flds[0].encode('utf8')]
            value = np.array(list(map(int, (flds[1].split(',')))))
            fvalue = np.array(list(map(float, (flds[2].split(',')))))
            example = tm.dataToExample([name, value, fvalue])
            writer.write(example.SerializeToString())

        except tf.errors.OutOfRangeError:
            break
        except Exception:
            print('a')
    writer.close()

    test_handle = sess.run(tf_iterator.string_handle())
    sess.run(tf_iterator.initializer)
    while True:
        try:
            b = sess.run(next_element, feed_dict={handle: test_handle})
            print(b['fvalue'])
        except tf.errors.OutOfRangeError:
            break
