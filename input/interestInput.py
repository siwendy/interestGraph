'''
Created on 2019年1月7日

@author: zhangyanqing1
'''
import tensorflow as tf

from utils.tfRecordManager import TFRecordConvertor
from utils.tfRecordManager import TFRecordWriter
from utils.tfRecordManager import TFRecordReader

FLAGS = tf.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


class InterestWriter(TFRecordWriter):
    def _lineToData(self, line):
        flds = line.strip().split('\t')
        label = int(flds[0])

        embedding = flds[1].split(',')
        embedding = list(map(int, embedding))
        return [[label], embedding]

    def _getConvertor(self):
        featureNames = ["label", "embedding"]
        featureTypes = [tf.int64, tf.float32]
        return TFRecordConvertor(featureNames, featureTypes)


class InterestReader(TFRecordReader):
    def _getConvertor(self):
        featureNames = ["label", "embedding"]
        featureTypes = [tf.int64, tf.float32]
        featureDims = [1, 2]
        return TFRecordConvertor(featureNames, featureTypes, featureDims)

    def _data_mapper(self, features):
        label = features['label']
        embedding = features['embedding']
        #embedding = tf.sparse_tensor_to_dense(features['embedding'])
        return embedding, label


def toTFRecord(dataPattern, tfRecordPattern):
    writer = InterestWriter(dataPattern,
                            tfRecordPattern,
                            fileLimit=2)
    writer.writeTFRecord()


def get_train_batch(filePattern, batchSize=None):
    dataReader = InterestReader(filePattern,
                                dataRepeat=300000,
                                batchSize=batchSize)
    data_iterator = dataReader.read()
    return data_iterator


def get_test_batch(filePattern, batchSize=None):
    dataReader = InterestReader(filePattern,
                                dataRepeat=10000,
                                batchSize=batchSize)
    data_iterator = dataReader.read()
    return data_iterator


def build_input():
    pass


if __name__ == '__main__':
    toTFRecord('../data/*.txt', '../data/tfRecordFile')

    train_iterator = get_train_batch('../data/*.tf', batchSize=20)
    test_iterator = get_test_batch('../data/*.tf', batchSize=20)

    handle = tf.placeholder(tf.string, shape=[])

    iterator = tf.data.Iterator.from_string_handle(
        handle,
        output_types=train_iterator.output_types,
        output_shapes=train_iterator.output_shapes,
        output_classes=train_iterator.output_classes)
    next_record = iterator.get_next()
    print(next_record)

    init = (tf.global_variables_initializer(),
            tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init)
        sess.run(train_iterator.initializer)
        sess.run(test_iterator.initializer)

        train_handle = sess.run(train_iterator.string_handle())
        test_handle = sess.run(test_iterator.string_handle())

        while True:
            try:
                e, l = sess.run(next_record, feed_dict={handle: train_handle})
                print('Train:')
                print(e, l)
                e, l = sess.run(next_record, feed_dict={handle: test_handle})
                print('Test:')
                print(e, l)
            except tf.errors.OutOfRangeError:
                break
