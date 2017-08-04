import tensorflow as tf
import numpy as np
from ops import *

BATCH_SIZE = 64


def generator(noise, yraw, train=True):
    with tf.variable_scope("generator") as scope:
        # reshape label data.
        # yb = tf.reshape(yraw, [BATCH_SIZE, 1, 1, 10], name='label')
        # concat noise to label. use the label as additional features. size is batch_size * (noise_col+label_col)
        noise = tf.concat([noise, yraw], 1, name="noise_concat_label")
        # To batch_size * 1024
        gFull1 = fully_connected(noise, 1024, 'g_fully_connected1')
        
        return noise


# label data.
# yraw = np.array([12, 11, 4, 6, 3, 4, 1, 4, 1, 1, 1, 5, 4, 3, 1, 4])
# rey = np.reshape(yraw, [4, 1, 1, 4])
# # noise data
# noi = np.array([[1,3,4],[2,3,4]])
# res=np.concatenate((noi,yraw),1)
# print(res)
