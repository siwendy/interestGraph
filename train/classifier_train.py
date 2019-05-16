# -*- coding: utf-8 -*-
# vim: set ts=2 sw=2 sts=2 tw=80 et:
# pylint: disable=missing-docstring
import sys
import tensorflow as tf

FLAGS = tf.flags.FLAGS
logger = tf.logging
logger.set_verbosity(tf.logging.INFO)

# Data Path
tf.flags.DEFINE_string(
    "training_data_pattern",
    "../data/*.tf", "training data FILE PATTERN")
tf.flags.DEFINE_string(
    "inner_validation_data_pattern",
    "inner_validation_data", "inner validation data FILE PATTERN")
tf.flags.DEFINE_string(
    "validation_data_pattern",
    "../data/*.tf", "validation data FILE PATTERN")
tf.flags.DEFINE_string(
    "model_dir", "./model", "model DIR")
tf.flags.DEFINE_string(
    "pretrained_model_dir",
    "pretrained_model", "pretrained model DIR")
tf.flags.DEFINE_string(
    "summaries_dir", "./eventLog", "summary DIR")

# Model Load and Save
tf.flags.DEFINE_integer(
    "load_epoch", 0, "load the model on an epoch using the model-prefix")
tf.flags.DEFINE_integer(
    "save_freq", 1000, "save model each number iterations")
tf.flags.DEFINE_integer(
    "ckpt_max_to_keep", 100,
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
    "cluster_num", 2,
    "validate model each number iterations")
tf.flags.DEFINE_integer(
    "title_embedding_size", 512, "title embedding size from input")
tf.flags.DEFINE_integer(
    "embedding_size", 10, "title embedding size output")
tf.flags.DEFINE_float(
    "lr", 0.1, "the initial learning rate")
tf.flags.DEFINE_bool(
    "enable_dropout", True, "whether to use dropout")

# Evaluation
tf.flags.DEFINE_bool(
    "enable_validation", True, "whether to use validation")
tf.flags.DEFINE_bool(
    "enable_inner_validation", False,
    "whether to use inner validation")
tf.flags.DEFINE_integer(
    "validation_data_size", 1024, "validation data size ")
tf.flags.DEFINE_integer(
    "validate_interval", 1000,
    "validate model each number iterations")

from input import interestInput

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
    if mode == 'train':
        regularization_losses = tf.get_collection(
            tf.GraphKeys.REGULARIZATION_LOSSES)
        if len(regularization_losses) > 0:
            logger.info(regularization_losses)
            loss = tf.add_n(regularization_losses) + loss
    return loss


class Train(object):
    def __init__(self, scope):
        with tf.variable_scope("train_validate", reuse=tf.AUTO_REUSE):
            train_iterator = interestInput.get_train_batch(
                FLAGS.training_data_pattern, FLAGS.batch_size)
            x, labels = train_iterator.get_next()

            # create model and loss function
            loss = model_with_loss(x, labels, "train")
            tf.summary.scalar('cross_entropy + regularization', loss)

            global_step = tf.train.get_or_create_global_step()
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
                # compute gradient
                optimizer = tf.train.GradientDescentOptimizer(FLAGS.lr)
                # optimizer = tf.train.AdamOptimizer(FLAGS.lr)
                grads_and_vars = optimizer.compute_gradients(loss)
                capped_grads_and_vars = grads_and_vars
                train_step = optimizer.apply_gradients(
                    capped_grads_and_vars, global_step=global_step)

        self._merged_summary = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        )
        self._train_iterator = train_iterator
        self._train_op = train_step
        self._global_step = global_step
        self._loss = loss
        self._x = x
        self._saver = tf.train.Saver(max_to_keep=FLAGS.ckpt_max_to_keep)

    def train_iter(self, sess):
        _, loss, step, summary = sess.run(
            [self._train_op, self._loss, self._global_step, self._merged_summary])
        return loss, step, summary

    def initialize(self, sess, save_path=None):
        init_ops = [tf.global_variables_initializer(),
                    tf.local_variables_initializer()]
        sess.run(init_ops)
        sess.run(self._train_iterator.initializer)
        self._train_handle = sess.run(self._train_iterator.string_handle())

        if save_path is not None:
            logger.info('loading model %s' % save_path)
            self._saver.restore(sess, save_path)

    def set_global_step(self, sess, step):
        set_op = tf.assign(self._global_step, step)
        sess.run(set_op)

    def save(self, sess, save_path):
        self._saver.save(sess, save_path)


class Validation(object):
    def __init__(self,
                 scope,
                 data_pattern=FLAGS.validation_data_pattern,
                 batch_size=FLAGS.batch_size,
                 name=None,):
        # create model
        with tf.variable_scope("train_validate", reuse=tf.AUTO_REUSE):
            test_iterator = interestInput.get_test_batch(
                data_pattern, batch_size)
            test_x, test_labels = test_iterator.get_next()

            out = model_with_loss(test_x, test_labels, "eval")
        self.loss, self.correct, self.result, self.labels = out
        self._test_iterator = test_iterator
        self._batch_size = batch_size
        if name is not None:
            tf.summary.scalar('%s_cross_entropy' % name, self.loss)
            tf.summary.scalar('%s_correct' % name, self.correct)
        self._merged_summary = tf.summary.merge(
            tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
        )

    def initialize(self, sess):
        sess.run(self._test_iterator.initializer)

    def validate(self, sess):
        sess.run(self._test_iterator.initializer)
        loss, correct, result, labels, summary = sess.run([self.loss, self.correct,
                                                           self.result, self.labels, self._merged_summary])
        logger.info('validation_data loss and correct:\t%f\t%d/%d' % (loss, correct,
                                                                      self._batch_size))
        logger.info('result and labels:\n%s\n%s' %
                    (str(result[:10]), str(labels[:10])))
        return summary


def print_trainable_variables(sess):
    logger.info("=== Trainable Variables ===")
    variables_names = [v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k, v in zip(variables_names, values):
        logger.info("Trainable Variable: %s" % k)
        logger.info("Shape: %s" % (str)(v.shape))


def main():
    config = tf.ConfigProto(allow_soft_placement=True,
                            log_device_placement=False)
    #    intra_op_parallelism_threads=16, inter_op_parallelism_threads=16)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # train
    with tf.device('/GPU:0'):
        with tf.name_scope("train") as train_scope:
            train = Train(train_scope)

    # validation
    if FLAGS.enable_validation:
        with tf.device('/CPU:0'):
            with tf.name_scope("out_test") as test_scope:
                outer_validation = Validation(
                    data_pattern=FLAGS.validation_data_pattern,
                    batch_size=FLAGS.validation_data_size,
                    name='outer_validation',
                    scope=test_scope)

    if FLAGS.enable_inner_validation:
        with tf.device('/CPU:0'):
            with tf.name_scope("inner_test") as test_scope:
                inner_validation = Validation(
                    data_ppattern=FLAGS.inner_validation_data_path,
                    batch_size=FLAGS.validation_data_size,
                    name='inner_validation',
                    scope=test_scope)

    # load epoch
    if bool(FLAGS.load_epoch):
        save_path = '%s/model-%d' % (FLAGS.pretrained_model_dir,
                                     FLAGS.load_epoch)
        train.initialize(sess, save_path)
        train.set_global_step(sess, FLAGS.load_epoch)
    else:
        train.initialize(sess)

    train_writer = tf.summary.FileWriter(
        FLAGS.summaries_dir, sess.graph)

    print_trainable_variables(sess)

    # start training
    try:
        while True:
            loss, step, summary = train.train_iter(sess)

            if step % 100 == 0:
                # write time spent
                logger.info("loss: %f | step: %d" % (loss, step))
                train_writer.add_summary(summary, step)

            if step % FLAGS.validate_interval == 0:
                logger.info("validation:")
                if FLAGS.enable_validation:
                    summary = outer_validation.validate(sess)
                    train_writer.add_summary(summary, step)
                if FLAGS.enable_inner_validation:
                    summary = inner_validation.validate(sess)
                    train_writer.add_summary(summary, step)

            if step % FLAGS.save_freq == 0:
                logger.info("save model...")
                train.save(sess, '%s/model-%d' % (FLAGS.model_dir, step))
    except tf.errors.OutOfRangeError:
        train.save(sess, '%s/model-%d' % (FLAGS.model_dir, step))
        logger.info("train finished.")

#     except Exception as e:
#         print(e)
#         train.save(sess, '%s/model-%d' % (FLAGS.model_dir, step))

    finally:
        sess.close()


if __name__ == "__main__":
    main()
