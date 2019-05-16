import tensorflow as tf
import numpy as np
import cPickle
import base64

FLAGS = tf.flags.FLAGS


def get_batch_csv(filenames, batch_size, line_len, epochs, num_threads=12):
    record_reader = tf.TextLineReader(name='record_reader')
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=False, num_epochs=epochs)
    _, records = record_reader.read_up_to(filename_queue, batch_size)
    # tensor = tf.stack(tf.decode_csv(records, [[0.0]]*line_len), 1)
    tensor = tf.stack(decode_numpy(records), 1)
    tensor = tf.cast(tensor, tf.float32)
    batch = tf.train.batch([tensor], batch_size=batch_size,
                           num_threads=num_threads, capacity=batch_size * 200,
                           enqueue_many=True)
    return batch


def get_batch_tfrecord(filenames, batch_size, epochs, num_threads=12):
    line_len = FLAGS.title_embedding_size + FLAGS.num_hard_negative + 1
    record_reader = tf.TFRecordReader(name='record_reader')
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=False, num_epochs=epochs)
    _, records = record_reader.read_up_to(filename_queue, batch_size)
    # tensor = tf.stack(tf.decode_csv(records, [[0.0]]*line_len), 1)
    tensor = tf.parse_example(
        records, features={'f': tf.FixedLenFeature(shape=line_len, dtype=tf.float32)})['f']
    random_negative = tf.random_uniform(shape=[tf.shape(tensor)[
                                        0], FLAGS.num_negative - FLAGS.num_hard_negative], dtype=tf.int32, minval=1, maxval=100001)
    random_negative = tf.cast(random_negative, tf.float32)
    # batch = tf.train.batch([tensor, random_negative], batch_size=batch_size,
    #     num_threads=num_threads, capacity=batch_size*200,
    #     enqueue_many=True)
    batch = tf.train.shuffle_batch([tensor, random_negative], min_after_dequeue=batch_size * 100, batch_size=batch_size,
                                   num_threads=num_threads, capacity=batch_size * 200,
                                   enqueue_many=True)
    batch = tf.concat(batch, 1)
    return batch


def get_vlen_batch(filenames, batch_size, epochs, num_threads=12):
    record_reader = tf.TFRecordReader(name='record_reader')
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=False, num_epochs=epochs)
    _, records = record_reader.read_up_to(filename_queue, batch_size)
    tensor = tf.parse_example(
        records, features={'f': tf.VarLenFeature(dtype=tf.float32)})['f']
    tensor = tf.sparse_tensor_to_dense(tensor)
    truncated_size = tf.minimum(
        tf.shape(tensor)[1] - 1, FLAGS.max_token_num * FLAGS.title_embedding_size)
    data = tf.slice(tensor, [0, 0], [-1, truncated_size])
    data = tf.pad(data, [[0, 0], [0, FLAGS.max_token_num *
                                  FLAGS.title_embedding_size - truncated_size]])
    label = tf.slice(tensor, [0, tf.shape(tensor)[1] - 1], [-1, 1])
    random_negative = tf.random_uniform(shape=[tf.shape(tensor)[
                                        0], FLAGS.num_negative - FLAGS.num_hard_negative], dtype=tf.int32, minval=1, maxval=100001)
    random_negative = tf.cast(random_negative, tf.float32)
    batch = tf.train.batch([data, label, random_negative],
                           batch_size=batch_size, num_threads=num_threads, capacity=batch_size * 200,
                           enqueue_many=True, dynamic_pad=True)
    batch = tf.concat(batch, 1)
    return batch


def get_all_tfrecord(filename, data_size):
    line_len = FLAGS.title_embedding_size + FLAGS.num_hard_negative + 1
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    all_data = []
    data_count = 0
    for string_record in record_iterator:
        data_count += 1
        if data_count > data_size:
            break
        tensor = tf.parse_single_example(
            string_record, features={'f': tf.FixedLenFeature(shape=line_len, dtype=tf.float32)})['f']
        random_negative = tf.random_uniform(
            shape=[FLAGS.num_negative - FLAGS.num_hard_negative], dtype=tf.int32, minval=1, maxval=100001)
        random_negative = tf.cast(random_negative, tf.float32)
        all_data.append(tf.concat([tensor, random_negative], 0))
    return tf.stack(all_data)


def get_all_vlen_tfrecord(filename, data_size):
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    all_data = []
    data_count = 0
    for string_record in record_iterator:
        data_count += 1
        if data_count > data_size:
            break
        tensor = tf.parse_single_example(
            string_record, features={'f': tf.VarLenFeature(dtype=tf.float32)})['f']
        tensor = tf.sparse_tensor_to_dense(tensor)
        truncated_size = tf.minimum(
            tf.shape(tensor)[0] - 1, FLAGS.max_token_num * FLAGS.title_embedding_size)
        data = tf.slice(tensor, [0], [truncated_size])
        data = tf.reshape(data, [1, -1])
        data = tf.pad(data, [[0, 0], [0, FLAGS.max_token_num *
                                      FLAGS.title_embedding_size - truncated_size]])
        data = tf.reshape(data, [-1])
        label = tf.slice(tensor, [tf.shape(tensor)[0] - 1], [1])
        random_negative = tf.random_uniform(
            shape=[FLAGS.num_negative - FLAGS.num_hard_negative], dtype=tf.int32, minval=1, maxval=100001)
        random_negative = tf.cast(random_negative, tf.float32)
        all_data.append(tf.concat([data, label, random_negative], 0))
    return tf.stack(all_data)


def get_batch(filenames, batch_size, epochs, num_threads=12, file_type='csv'):
    line_len = FLAGS.title_embedding_size + FLAGS.num_negative + 1
    if file_type == 'csv':
        return get_batch_csv(filenames, batch_size, line_len, epochs, num_threads)
    else:
        return get_batch_tfrecord(filenames, batch_size, epochs, num_threads)


def generate_batch_for_classifier(filenames, batch_size, epochs, num_threads=12):
    line_len = FLAGS.title_embedding_size + FLAGS.num_hard_negative + 1
    record_reader = tf.TFRecordReader(name='record_reader')
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=False, num_epochs=epochs)
    _, records = record_reader.read_up_to(filename_queue, batch_size)
    # tensor = tf.stack(tf.decode_csv(records, [[0.0]]*line_len), 1)
    tensor = tf.parse_example(
        records, features={'f': tf.FixedLenFeature(shape=line_len, dtype=tf.float32)})['f']
    batch = tf.train.shuffle_batch([tensor],
                                   min_after_dequeue=batch_size * 100, batch_size=batch_size,
                                   num_threads=num_threads, capacity=batch_size * 200,
                                   enqueue_many=True)
    data = tf.slice(batch, [0, 0], [-1, FLAGS.title_embedding_size])
    labels = tf.cast(
        tf.slice(batch, [0, FLAGS.title_embedding_size], [-1, 1]), tf.int32) - 1
    return data, labels


def generate_batch_for_matrix_classifier(filenames, batch_size, epochs, num_threads=12):
    line_len = FLAGS.title_embedding_size + FLAGS.num_hard_negative + 1
    record_reader = tf.TFRecordReader(name='record_reader')
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=False, num_epochs=epochs)
    _, records = record_reader.read_up_to(filename_queue, batch_size)
    # tensor = tf.stack(tf.decode_csv(records, [[0.0]]*line_len), 1)
    tensor = tf.parse_example(
        records, features={'f': tf.VarLenFeature(dtype=tf.float32)})['f']
    batch = tf.train.batch([tensor],
                           batch_size=batch_size, num_threads=num_threads, capacity=batch_size * 200,
                           enqueue_many=True, dynamic_pad=True)
    data = tf.slice(batch, [0, 0], [-1, FLAGS.title_embedding_size])
    labels = tf.cast(
        tf.slice(batch, [0, FLAGS.title_embedding_size], [-1, 1]), tf.int32) - 1
    return data, labels


def generate_batch_for_topk(filenames, batch_size, epochs, num_threads=12):
    line_len = FLAGS.title_embedding_size + FLAGS.num_hard_negative + 1
    record_reader = tf.TFRecordReader(name='record_reader')
    filename_queue = tf.train.string_input_producer(
        filenames, shuffle=False, num_epochs=epochs)
    _, records = record_reader.read_up_to(filename_queue, batch_size)
    # tensor = tf.stack(tf.decode_csv(records, [[0.0]]*line_len), 1)
    tensor = tf.parse_example(
        records, features={'f': tf.FixedLenFeature(shape=line_len, dtype=tf.float32)})['f']
    data = tf.slice(tensor, [0, 0], [-1, FLAGS.title_embedding_size + 1])
    random_negative = tf.random_uniform(shape=[tf.shape(
        data)[0], FLAGS.num_negative], dtype=tf.int32, minval=1, maxval=100001)
    random_negative = tf.cast(random_negative, tf.float32)
    batch = tf.train.shuffle_batch([data, random_negative],
                                   min_after_dequeue=batch_size * 100, batch_size=batch_size,
                                   num_threads=num_threads, capacity=batch_size * 200,
                                   enqueue_many=True)
    batch = tf.concat(batch, 1)
    return batch


def generate_all_batch_for_classifier(filename, data_size):
    line_len = FLAGS.title_embedding_size + FLAGS.num_hard_negative + 1
    record_iterator = tf.python_io.tf_record_iterator(path=filename)
    all_data = []
    labels = []
    data_count = 0
    for string_record in record_iterator:
        data_count += 1
        if data_count > data_size:
            break
        tensor = tf.parse_single_example(
            string_record, features={'f': tf.FixedLenFeature(shape=line_len, dtype=tf.float32)})['f']
        data = tf.slice(tensor, [0], [FLAGS.title_embedding_size])
        label = tf.cast(
            tf.slice(tensor, [FLAGS.title_embedding_size], [1]), tf.int32) - 1
        all_data.append(data)
        labels.append(label)
    return tf.stack(all_data), tf.stack(labels)
