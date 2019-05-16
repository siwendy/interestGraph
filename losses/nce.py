import tensorflow as tf
import math

def loss(inputs, labels, num_classes, num_sampled, embedding_size, mode="train"):

  # Construct the variables for the NCE loss
  with tf.variable_scope('loss', reuse=tf.AUTO_REUSE) as scope:
    with tf.name_scope('weights'):
      nce_weights = tf.get_variable(
          'nce_weight',
          shape=[num_classes, embedding_size],
          initializer=tf.truncated_normal_initializer(
              stddev=1.0 / math.sqrt(embedding_size)))
    with tf.name_scope('biases'):
      nce_biases = tf.get_variable('nce_bias',
          shape=[num_classes],
          initializer=tf.zeros_initializer())

  with tf.name_scope('loss'):
    if mode == "train":
      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      # Explanation of the meaning of NCE loss:
      #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
      loss = tf.reduce_mean(
          tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=labels,
              inputs=inputs,
              num_sampled=num_sampled,
              num_classes=num_classes,
              partition_strategy="div"))
      tf.summary.scalar('train_loss', loss)
      return loss
    elif mode == "eval":
      logits = tf.matmul(inputs, tf.transpose(nce_weights))
      logits = tf.nn.bias_add(logits, nce_biases)

      labels_one_hot = tf.one_hot(tf.reshape(labels, [-1]), num_classes)
      loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)
      loss = tf.reduce_sum(loss, axis=1)
      loss = tf.reduce_mean(loss)

      tf.summary.scalar('eval_loss', loss)

      # For a classifier model, we can use the in_top_k Op.
      # It returns a bool tensor with shape [batch_size] that is true for
      # the examples where the label is in the top k (here k=1)
      # of all logits for that example.
      _, result = tf.nn.top_k(logits, 10)
      correct = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, tf.reshape(labels, [-1]), 1), tf.int32))
      tf.summary.scalar('eval_correct', correct)
      # Return the number of true entries.
      return loss, correct, result, labels
    elif mode == "test":
      logits = tf.matmul(inputs, tf.transpose(nce_weights))
      logits = tf.nn.bias_add(logits, nce_biases)
      value, result = tf.nn.top_k(logits, 100)
      correct = tf.reduce_sum(tf.cast(tf.nn.in_top_k(logits, labels, 5), tf.int32))
      # Return the number of true entries.
      return result, value, correct

