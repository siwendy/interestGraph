# coding=utf-8
'''
Created on 2018年10月30日

@author: zhangyanqing1
'''
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string("tf_filename", "interest_distribution",
                    "interest distribution")
flags.DEFINE_integer("title_embedding_fld", 6, "embedding field")
flags.DEFINE_integer("num_negative", 0, "negative sample number")
flags.DEFINE_integer("label_fld", 0, "label field")
flags.DEFINE_boolean("enable_resample", False, "enable resample")
flags.DEFINE_boolean("enable_shuffle", False, "enable shuffle")
infiles = []
# infiles.append("example.tfrecords")
infiles.append("train-00000-of-00010")


def parse2dense(parse):
    return tf.sparse_to_dense(parse.indices, parse.dense_shape, parse.values, 5)


def main():
    pass


if __name__ == '__main__':
    reader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(infiles)
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'decode_post': tf.VarLenFeature(tf.int64),
            'decode_pre': tf.VarLenFeature(tf.int64),
            'encode': tf.VarLenFeature(tf.int64)
        })
    encode = tf.cast(features['encode'], tf.int32)
    pre = tf.cast(features['decode_pre'], tf.int32)
    post = tf.cast(features['decode_post'], tf.int32)
    encode = parse2dense(encode)
    pre = parse2dense(pre)
    post = parse2dense(post)

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        for i in range(100):
            decode_pre, decode_label, encode_now = sess.run(
                [pre, post, encode])
            # print(type(image))
            #tf.sparse_to_dense(sparse_indices, output_shape, sparse_values, default_value, validate_indices, name)
            #a = tf.sparse_to_dense(image.indices,image.dense_shape, image.values,0)
            # print(a)
            print('encode:', encode_now)
            print('pre:', decode_pre)
            print('post:', decode_label)
