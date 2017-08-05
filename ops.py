import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    # what's different between training and learnig.
    res = tf.get_variable(name, shape, trainable=trainable,
                          initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))
    return res


def bias(name, shape, biasesStart=0.02, trainable=True):
    dtype = tf.float32
    res = tf.get_variable(name, shape, trainable=trainable,
                          initializer=tf.constant_initializer(biasesStart, dtype=dtype))
    return res


def fully_connected(value, outputSize, name, with_w=False):
    """
    value - [raw, column]. fully_connected/weights
    """
    shape = value.get_shape().as_list()

    with tf.variable_scope(name):
        weights = weight("weights", [shape[1], outputSize], 0.02)
        biases = bias("biases", [outputSize], 0.0)

    if with_w:
        return tf.add(tf.matmul(value, weights), biases), weights, biases
    else:
        return tf.matmul(value, weights) + biases


def batch_norm_layer(value, isTrain=True, name="batch_norm"):
    with tf.variable_scope(name) as scope:
        if isTrain:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True, is_training=isTrain, updates_collections=None, scope=scope)
        else:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True,
                              is_training=isTrain, reuse=True,
                              updates_collections=None, scope=scope)


# Concat feature and condition.
def conv_cond_concat(value, cond, name='concat'):
    value_shape = value.get_shape().as_list()
    cond_shape = cond.get_shape().as_list()
    # value_shape[0:3] is feature columns. cond_shape[3:] is condition's label column
    # expand cond to [batch_size, feature_size1, feature_size2, label_column]
    # add label column to each feature.
    with tf.variable_scope(name):
        return tf.concat([value, cond * tf.ones(value_shape[0:3] + cond_shape[3:])], 3)


def deconv2d(value, output_shape, k_h=5, k_w=5, strides=[1, 2, 2, 1], name="deconv2d", with_w=False):
    with tf.variable_scope(name) as scope:
        weights = weight(
            'weights', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides=strides)

        biases = bias("biases", [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv

