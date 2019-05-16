# -*- coding: utf-8 -*-
# vim: set ts=2 sw=2 sts=2 tw=80 et:
# pylint: disable=missing-docstring

"""Train the mv-dssm model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import sys
import os

import numpy as np
import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Data Path
tf.flags.DEFINE_string(
    "training_data_path",
    "training_data", "training data DIR")
tf.flags.DEFINE_string(
    "inner_validation_data_path",
    "inner_validation_data", "inner validation data DIR")
tf.flags.DEFINE_string(
    "validation_data_path",
    "validation_data", "validation data DIR")
tf.flags.DEFINE_string(
    "model_dir", "model", "model DIR")
tf.flags.DEFINE_string(
    "pretrained_model_dir",
    "pretrained_model", "pretrained model DIR")
tf.flags.DEFINE_string(
    "summaries_dir", "eventLog", "summary DIR")

# Model Load and Save
tf.flags.DEFINE_integer(
    "load_epoch", 0, "load the model on an epoch using the model-prefix")
tf.flags.DEFINE_integer(
    "save_freq", 10000, "save model each number iterations")
tf.flags.DEFINE_integer(
    "ckpt_max_to_keep", 10,
    "maximum number of recent checkpoint to keep")

# Model
tf.flags.DEFINE_string(
    "network", "mlp", "the network to use (simple, mlp)")
tf.flags.DEFINE_string(
    "loss", "softmax", "the loss to use (softmax, nce)")

# Hyper Param.
tf.flags.DEFINE_integer(
    "batch_size", 128, "batch size ")
tf.flags.DEFINE_integer(
    "num_epochs", 10, "training epochs")
tf.flags.DEFINE_integer(
    "num_negative", 10, "number of negative samples per entry")
tf.flags.DEFINE_integer(
    "num_hard_negative", 0,
    "number of hard negative samples per entry")
tf.flags.DEFINE_integer(
    "cluster_num", 100000,
    "validate model each number iterations")
tf.flags.DEFINE_integer(
    "title_embedding_size", 512, "title embedding size")
tf.flags.DEFINE_integer(
    "embedding_size", 128, "title embedding size")
tf.flags.DEFINE_float(
    "lr", 0.1, "the initial learning rate")
tf.flags.DEFINE_bool(
    "enable_dropout", False, "whether to use dropout")

# Evaluation
tf.flags.DEFINE_bool(
    "enable_validation", False, "whether to use validation")
tf.flags.DEFINE_bool(
    "enable_inner_validation", False,
    "whether to use inner validation")
tf.flags.DEFINE_integer(
    "validation_data_size", 1024, "validation data size ")
tf.flags.DEFINE_integer(
    "validate_interval", 10000,
    "validate model each number iterations")

from mv_data import generate_batch_for_classifier
from mv_data import generate_all_batch_for_classifier

import importlib
network = importlib.import_module('models.' + FLAGS.network).network
loss_module = importlib.import_module('losses.' + FLAGS.loss)
loss_fun = loss_module.loss


def model_with_loss(data, labels, mode):
    embed = network(
        inputs=data,
        embedding_size=FLAGS.embedding_size,
        mode=mode,
        enable_dropout=True)
    loss = loss_fun(
        inputs=embed,
        labels=labels,
        num_classes=FLAGS.cluster_num,
        trainable=True,
        mode=mode)

    regularization_losses = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = tf.add_n(regularization_losses) + loss
    return loss


class Train(object):
    def __init__(self):
        # train and validate
        with tf.variable_scope("train_validate"):
            if os.path.isdir(FLAGS.training_data_path):
                file_name_list = [FLAGS.training_data_path + '/' + file_name for file_name in
                                  os.listdir(FLAGS.training_data_path)]
            else:
                file_name_list = FLAGS.training_data_path.split(",")
            x, labels = generate_batch_for_classifier(
                file_name_list, FLAGS.batch_size, FLAGS.num_epochs)
            print('Inputs Shape: ', x.shape)

            # create model and loss function
            loss = model_with_loss(x, labels, "train")
            tf.summary.scalar('cross_entropy+regularization', loss)

            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                # compute gradient
                optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
                # optimizer = tf.train.AdamOptimizer(FLAGS.lr)
                grads_and_vars = optimizer.compute_gradients(loss)
                capped_grads_and_vars = grads_and_vars
                #capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]
                train_step = optimizer.apply_gradients(capped_grads_and_vars)
                # gradients, variables = zip(*optimizer.compute_gradients(cost))
                # gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.clip)
                # optim = optimizer.apply_gradients(zip(gradients, variables))

            self._train_op = train_step
            self._loss = loss
            self._x = x
            self._saver = tf.train.Saver(max_to_keep=FLAGS.ckpt_max_to_keep)

    def train_iter(self, sess):
        l, _ = sess.run([self._loss, self._train_op])
        return l

    def initialize(self, sess, save_path=None):
        init_ops = [tf.global_variables_initializer(),
                    tf.local_variables_initializer()]
        sess.run(init_ops)

        if save_path is not None:
            print('loading model %s' % save_path)
            self._saver.restore(sess, save_path)

    def save(self, sess, save_path):
        self._saver.save(sess, save_path)


class Validation(object):
    def __init__(self, data_path, data_size=10, name=None):
        """
        Args:
          data_path: data path
          data_size: data size
        """
        self.data_size = data_size
        x, labels = generate_all_batch_for_classifier(data_path, data_size)
        # create model
        with tf.variable_scope("train_validate"):
            out = model_with_loss(x, labels, "eval")
            self.loss, self.correct, self.result, self.labels = out

        if name is not None:
            tf.summary.scalar('%s_cross_entropy' % name, self.loss)
            tf.summary.scalar('%s_correct' % name, self.correct)

    def validate(self, sess):
        """
        Args:
          sess: tf session
        """
        loss, correct, result, labels = sess.run([self.loss, self.correct,
                                                  self.result, self.labels])
        print('validation_data loss and correct:\t%f\t%d/%d' % (loss, correct,
                                                                self.data_size))
        print('result and labels:\t%s\t%s' %
              (str(result[:10]), str(labels[:10])))
        #print('logits and inputs:\t%s\t%s' % (str(logits[:10]), str(inputs[:10])))


def print_trainable_variables(sess):
    print("=== Trainable Variables ===")
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        print("Trainable Variable: ", k)
        print("Shape: ", v.shape)


def main():
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    #    intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # train
    train = Train()

    # validation
    if FLAGS.enable_validation:
        outer_validation = Validation(
            data_path=FLAGS.validation_data_path,
            data_size=FLAGS.validation_data_size,
            name='outer_validation')

    if FLAGS.enable_inner_validation:
        inner_validation = Validation(
            data_path=FLAGS.inner_validation_data_path,
            data_size=FLAGS.validation_data_size,
            name='inner_validation')

    # load epoch
    num_iter = 0
    if bool(FLAGS.load_epoch):
        save_path = '%s/model-%d' % (FLAGS.pretrained_model_dir,
                                     FLAGS.load_epoch)
        train.initialize(sess, save_path)
        num_iter = FLAGS.load_epoch
    else:
        train.initialize(sess)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir, sess.graph)

    print_trainable_variables(sess)

    # start training
    try:
        sum_loss = 0
        while True:
            num_iter += 1
            l = train.train_iter(sess)
            sum_loss += l

            print_interval = 100
            if num_iter % print_interval == 0:
                # write time spent
                sys.stdout.write("loss: %f | iter: %d\n" %
                                 (sum_loss / print_interval, num_iter))
                sys.stdout.flush()
                sum_loss = 0
                summary, = sess.run([merged])
                train_writer.add_summary(summary, num_iter)

            if num_iter % FLAGS.validate_interval == 0:
                print("validation:")
                if FLAGS.enable_validation:
                    outer_validation.validate(sess)
                if FLAGS.enable_inner_validation:
                    inner_validation.validate(sess)

            if num_iter % FLAGS.save_freq == 0:
                print("save model")
                train.save(sess, '%s/model-%d' % (FLAGS.model_dir, num_iter))

    except Exception as e:
        train.save(sess, '%s/model-%d' % (FLAGS.model_dir, num_iter))
        coord.request_stop(e)

    finally:
        train.save(sess, '%s/model-%d' % (FLAGS.model_dir, num_iter))
        coord.request_stop()
        coord.join(threads)


if __name__ == "__main__":
    main()
