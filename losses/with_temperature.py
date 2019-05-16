import tensorflow as tf

def distance(x, is_training):
  t = tf.slice(x, [0, 1, 0], [-1, -1, -1])
  t = tf.nn.l2_normalize(t, 2)
  q = tf.slice(x, [0, 0, 0], [-1, 1, -1])
  q = tf.nn.l2_normalize(q, 2)
  q = tf.tile(q, tf.stack([1, tf.shape(t)[1], 1]))
  pred = tf.reduce_sum(t * q, 2)
  return pred

def loss(pred):
  with tf.variable_scope("loss", reuse=tf.AUTO_REUSE) as scope:
    gamma = tf.get_variable("loss_gamma", initializer=1., trainable=True)
    # gamma = tf.get_variable("loss_gamma", initializer=10., trainable=False)
    pred = pred * gamma
    label = tf.zeros(tf.stack([tf.shape(pred)[0]]), dtype=tf.int32)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    loss = tf.reduce_mean(loss)
    return loss
