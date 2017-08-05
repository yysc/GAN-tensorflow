import tensorflow as tf
import numpy as np
from ops import *

BATCH_SIZE = 64


def generator(noise, yraw, train=True):
    with tf.variable_scope("generator") as scope:
        # concat noise to label. use the label as additional features. size is
        # batch_size * (noise_col+label_col)
        noise = tf.concat([noise, yraw], 1, name="noise_concat_label")
        # To batch_size * 1024
        gFull1 = fully_connected(noise, 1024, 'g_fully_connected1')
        h1 = tf.nn.relu(batch_norm_layer(gFull1, isTrain=train))
        h1 = tf.concat([h1, yraw], 1, name="active1_concat_yraw")
        # batch_size * 1034(include 10 columns of yraw) to batch_size * 128*49
        h2 = tf.nn.relu(batch_norm_layer(fully_connected(
            h1, 128 * 49, 'g_full_connected2'), isTrain=train, name="g_bn2"))
        # batch_size*128*49 to [batch_size, 7, 7, 128] why 128*49
        # batch_size * [[[128 dimension][]...[]],[[128
        # dimension][]...[]]...[[128 dimension][]...[]]]
        h2 = tf.reshape(h2, [BATCH_SIZE, 7, 7, 128], name="h2_reshape")
        # reshape label data. batch_size * [[[10 dimensions]]]
        yb = tf.reshape(yraw, [BATCH_SIZE, 1, 1, 10], name='label')
        # concat h2 to resize label. [batch_size, 7,7, 138]
        h2 = conv_cond_concat(h2, yb, name='active2_concat_yraw')
        # change pooling to conv.
        h3 = tf.nn.relu(batch_norm_layer(deconv2d(
            h2, [64, 14, 14, 128], name='g_deconv2d3'), isTrain=train, name='g_bn3'))
        h3 = conv_cond_concat(h3, yb, name='active3_concat_y')
        h4 = tf.nn.sigmoid(
            deconv2d(h3, [64, 28, 28, 1], name='g_deconv2d4'), name='generate_image')

        return h4


def discriminator(image, y, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            tf.get_variable_scope().reuse_variables()
        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='yb')
        x = conv_cond_concat(image, yb, name='image_concat_yb')
        conv1 = conv2d(x, 11, name='d_conv2d1')
        h1 = lrelu(conv1, name='lrelu1')
        h1 = conv_cond_concat(h1, yb, name='active1_concat_yb')

        h2 = lrelu(batch_norm_layer(
            conv2d(h1, 74, name='d_conv2d2'), name='d_bn2'), name="lrelu2")
        h2 = tf.reshape(h2, [BATCH_SIZE, -1], name='reshape_lrelu2_to_2d')
        h2 = tf.concat([h2, y], 1, name='lrelu2_concat_y')


def conv2d(value, output_dim, k_h=5, k_w=5, strides=[1, 2, 2, 1], name='conv2d'):
    with tf.variable_scope(name) as scope:
        weights = weight(
            'weights', [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        temp = tf.nn.bias_add(conv, biases)
        print(temp.shape)
        conv = tf.reshape(temp, conv.get_shape())
        print(conv.shape)
        return conv


def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name) as scope:
        return tf.maximum(x, leak * x, name=name)

# label data.
# yraw = np.array([12, 11, 4, 6, 3, 4, 1, 4, 1, 1, 1, 5, 4, 3, 1, 4])
# rey = np.reshape(yraw, [4, 1, 1, 4])
# # noise data
# noi = np.array([[1,3,4],[2,3,4]])
# res=np.concatenate((noi,yraw),1)
# print(res)
