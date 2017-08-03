import tensorflow as tf
from opsGan import *

BATCH_SIZE = 64

# Define generator. Generate pic from random noize and condition.


def generator(z, y, train=True):
    with tf.variable_scope("generator") as scope:
        # y is [BATCH_SIZE, 10]. transform y to four demension.
        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='yb')
        # concat restruction y and  noise z.
        z = tf.concat([z, y], 1, name='z_concat_y')
        # batch normalization and full connected layer.
        h1 = tf.nn.relu(batch_norm_layer(fully_connected(z, 1024, 'g_fully_connected1'),
                                        is_train=train, name='g_bn1'))
        # concat restruction and last layer.
        h1 = tf.concat([h1, y], 1, name='active1_concat_y')
        # full connected and relu.
        h2 = tf.nn.relu(batch_norm_layer(fully_connected(h1, 128 * 49, 'g_fully_connected2'),
                                        is_train=train, name='g_bn2'))
        h2 = tf.reshape(h2, [64, 7, 7, 128], name='h2_reshape')
        # Concat restruction and last layer.
        h2 = conv_cond_concat(h2, yb, name='active2_concat_y')

        h3 = tf.nn.relu(batch_norm_layer(deconv2d(h2, [64, 14, 14, 128],
                                                name='g_deconv2d3'),
                                        is_train=train, name='g_bn3'))
        h3 = conv_cond_concat(h3, yb, name='active3_concat_y')

        # sigmoid to normalize it in 0 to 1.
        h4 = tf.nn.sigmoid(deconv2d(h3, [64, 28, 28, 1],
                                    name='g_deconv2d4'), name='generate_image')

        return h4

#   Define discriminator.


def discriminator(image, y, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        # both real image and genearted image will go though discriminator.
        if reuse:
            tf.get_variable_scope().reuse_variables()

        # Concat the restruction condition.
        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='yb')
        x = conv_cond_concat(image, yb, name='image_concat_y')

        # conv, fire and concat restructions.
        h1 = lrelu(conv2d(x, 11, name='d_conv2d1'), name='lrelu1')
        h1 = conv_cond_concat(h1, yb, name='h1_concat_yb')

        # conv, fire and concat restructions.
        h2 = lrelu(batch_norm_layer(conv2d(h1, 74, name='d_conv2d2'),
                                    name='d_bn2'), name='lrelu2')
        h2 = tf.reshape(h2, [BATCH_SIZE, -1], name='reshape_lrelu2_to_2d')
        h2 = tf.concat([h2, y],1, name='lrelu2_concat_y')
        
        # fullly connected layer, batch_normal and relu. Concat condition.
        h3 = lrelu(batch_norm_layer(fully_connected(h2, 1024, name='d_fully_connected3'),
                                    name='d_bn3'), name='lrelu3')
        h3 = tf.concat([h3, y],1, name='lrelu3_concat_y')

        # Fully connected layer. sigmoid to 0-1.
        h4 = fully_connected(h3, 1, name='d_result_withouts_sigmoid')

        return tf.nn.sigmoid(h4, name='discriminator_result_with_sigmoid'), h4

# Define sample function.
def sampler(z, y, train=True):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, train=train)
