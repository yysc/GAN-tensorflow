import tensorflow as tf
import numpy as np
from ops import *

BATCH_SIZE = 64


def generator(noise, yraw, train=True):
    with tf.variable_scope("generator") as scope:
        # concat noise to label. use the label as additional features. size is batch_size * (noise_col+label_col)
        noise = tf.concat([noise, yraw], 1, name="noise_concat_label")
        # To batch_size * 1024
        gFull1 = fully_connected(noise, 1024, 'g_fully_connected1')
        h1 = tf.nn.relu(batch_norm_layer(gFull1, isTrain=train))
        h1 = tf.concat([h1, yraw], 1, name="active1_concat_yraw")
        # batch_size * 1034(include 10 columns of yraw) to batch_size * 128*49
        h2 = tf.nn.relu(batch_norm_layer(fully_connected(
            h1, 128 * 49, 'g_full_connected2'), isTrain=train, name="g_bn2"))
        # batch_size*128*49 to [batch_size, 7, 7, 128] why 128*49
        # batch_size * [[[128 dimension][]...[]],[[128 dimension][]...[]]...[[128 dimension][]...[]]]
        h2 = tf.reshape(h2, [BATCH_SIZE, 7, 7, 128], name="h2_reshape")
        # reshape label data. batch_size * [[[10 dimensions]]]
        yb = tf.reshape(yraw, [BATCH_SIZE, 1, 1, 10], name='label')
        # 
        h2 = conv_cond_concat(h2, yb, name='active2_concat_yraw')

        # h3 = tf.nn.relu(batch_norm_layer())

        return noise


# label data.
# yraw = np.array([12, 11, 4, 6, 3, 4, 1, 4, 1, 1, 1, 5, 4, 3, 1, 4])
# rey = np.reshape(yraw, [4, 1, 1, 4])
# # noise data
# noi = np.array([[1,3,4],[2,3,4]])
# res=np.concatenate((noi,yraw),1)
# print(res)
