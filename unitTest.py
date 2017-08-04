from read_data import read_data
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import batch_norm
from ganModel import BATCH_SIZE
import os
import numpy as np
from utils import save_images

# dataX, dataY = read_data()
# print("dataX size is ", dataX.shape," and data y is ", dataY.shape)


def generator(z, y, train=True):
    with tf.variable_scope("generator") as scope:
        # label data y.
        yb = tf.reshape(y, [BATCH_SIZE, 1, 1, 10], name='yb')
        # Noise z. Concat condition to noise raw data to expand features.
        # Concat on column dimension. the other dimension must be the same.
        z = tf.concat([z, y], 1, name='z_concat_y')
        # hidden layer 1
        full1 = fully_connected(z, 1024, 'g_fully_connected1')
        batch1 = batch_norm_layer(full1, isTrain=True, name='g_batch_norm1')
        h1 = tf.nn.relu(batch1)
        h1 = tf.concat([h1, y], 1, name="hidden1_concat_y")

        h2 = tf.nn.relu(batch_norm_layer(fully_connected(h1, 128 * 49, 'g_fully_connected2'), isTrain=train, name='g_batch_norm2'))
        h2 = tf.reshape(h2, [64, 7, 7, 128], name='h2_reshape')
        h2 = conv_cond_concat(h2, yb, name='hidden2_concat_y')
        h3 = tf.nn.relu(batch_norm_layer(deconv2d(
            h2, [64, 14, 14, 128], name='g_deconv2d3'), isTrain=train, name='g_batch_n3'))
        h3 = conv_cond_concat(h3, yb, name='hidden3_concat_y')
        h4 = tf.nn.sigmoid(
            deconv2d(h3, [64, 28, 28, 1], name='g_deconv2d4'), name='generate_image')

        return h4


def deconv2d(value, output_shape, k_h=5, k_w=5, strides=[1, 2, 2, 1], name='deconv2d', with_w=False):
    with tf.variable_scope(name) as scope:
        weights = weight(
            'weights', [k_h, k_w, output_shape[-1], value.get_shape()[-1]])
        deconv = tf.nn.conv2d_transpose(
            value, weights, output_shape, strides=strides)

        biases = bias('biases', [output_shape[-1]])
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        if with_w:
            return deconv, weights, biases
        else:
            return deconv


def conv_cond_concat(value, cond, name='concat'):
    value_shapes = value.get_shape().as_list()
    cond_shapes = cond.get_shape().as_list()
    with tf.variable_scope(name) as scope:
        return tf.concat([value, cond * tf.ones(value_shapes[0:3] + cond_shapes[3:])], 3)


def weight(name, shape, stddev=0.02, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.random_normal_initializer(stddev=stddev, dtype=dtype))
    return var


def bias(name, shape, bias_start=0.0, trainable=True):
    dtype = tf.float32
    var = tf.get_variable(name, shape, tf.float32, trainable=trainable,
                          initializer=tf.constant_initializer(bias_start, dtype=dtype))
    return var


def batch_norm_layer(value, isTrain=True, name='batch_norm'):
    with tf.variable_scope(name) as scope:
        if isTrain:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True, is_training=isTrain, updates_collections=None, scope=scope)
        else:
            return batch_norm(value, decay=0.9, epsilon=1e-5, scale=True, is_training=isTrain, reuse=True, updates_collections=None, scope=scope)


def fully_connected(value, output_shape, name='fully_connected', with_w=False):
    shape = value.get_shape().as_list()
    with tf.variable_scope(name):
        weights = weight('weights', [shape[1], output_shape], 0.02)
        biases = bias('biases', [output_shape], 0.0)

    if with_w:
        return tf.matmul(value, weights) + biases, weights, biases
    else:
        return tf.matmul(value, weights) + biases

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

def lrelu(x, leak=0.2, name='lrelu'):
    
    with tf.variable_scope(name) as scope:
        return tf.maximum(x, leak * x, name=name)

# ReLu layer.
def relu(value, name='relu'):
    with tf.variable_scope(name) as scope:
        return tf.nn.relu(value)


def conv2d(value, output_dim, k_h=5, k_w=5,
           strides=[1, 2, 2, 1], name='conv2d'):

    with tf.variable_scope(name) as scope:
        weights = weight('weights',
                         [k_h, k_w, value.get_shape()[-1], output_dim])
        conv = tf.nn.conv2d(value, weights, strides=strides, padding='SAME')
        biases = bias('biases', [output_dim])
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

        return conv

def sampler(z, y, train=True):
    tf.get_variable_scope().reuse_variables()
    return generator(z, y, train=train)

def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)

    train_dir = '/Users/yy/Documents/github/DCGAN-tensorflow/data/mnist/log'
    # label
    y = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name="y")
    # noise
    z = tf.placeholder(tf.float32, [None, 100], name="z")
    # real image
    images = tf.placeholder(
        tf.float32, [BATCH_SIZE, 28, 28, 1], name="real_images")

    # create generator. all generator variables are start with 'g_'
    genNet = generator(z, y)
    # create discriminator for real images. all discriminator variables are start with 'd_'
    D, D_logits  = discriminator(images, y,reuse = False)
    
    D_, D_logits_ = discriminator(genNet, y, reuse = True)
    
    # discriminator loss for real images.
    d_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits, labels=tf.ones_like(D)))
    # discrimminator loss for generated images.
    d_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.zeros_like(D_)))
    d_loss = d_loss_real + d_loss_fake
    g_loss = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_, labels=tf.ones_like(D_)))

    z_sum = tf.summary.histogram("z", z)
    d_sum = tf.summary.histogram("d", D)
    d__sum = tf.summary.histogram("d_", D_)
    G_sum = tf.summary.image("G", genNet)

    d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
    d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)
    d_loss_sum = tf.summary.scalar("d_loss", d_loss)                                                
    g_loss_sum = tf.summary.scalar("g_loss", g_loss)
    
    g_sum = tf.summary.merge([z_sum, d__sum, G_sum, d_loss_fake_sum, g_loss_sum])
    d_sum = tf.summary.merge([z_sum, d_sum, d_loss_real_sum, d_loss_sum])

    # all variables.
    t_vars = tf.trainable_variables()
    # get all discriminator variables.
    d_vars = [var for var in t_vars if 'd_' in var.name]
    # get all generator variables.
    g_vars = [var for var in t_vars if 'g_' in var.name]

    saver = tf.train.Saver()

    d_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1 = 0.5) \
               .minimize(d_loss, var_list = d_vars, global_step = global_step)
    g_optim = tf.train.AdamOptimizer(learning_rate=0.0002, beta1 = 0.5) \
                .minimize(g_loss, var_list = g_vars, global_step = global_step)

    samples = sampler(z, y)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.2
    sess = tf.InteractiveSession(config=config)

    init = tf.initialize_all_variables()   
    writer = tf.summary.FileWriter(train_dir, sess.graph)
    
    # read data and reshape.
    data_x, data_y = read_data()
    sample_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))
#    sample_images = data_x[0: 64]
    sample_labels = data_y[0: 64]
    sess.run(init)    


    # training 25 epochs
    for epoch in range(25):
        batch_idxs = 1093
        for idx in range(batch_idxs):        
            batch_images = data_x[idx*64: (idx+1)*64]
            batch_labels = data_y[idx*64: (idx+1)*64]
            batch_z = np.random.uniform(-1, 1, size=(BATCH_SIZE, 100))            
            
            # update discrimination parameters.
            _, summary_str = sess.run([d_optim, d_sum], 
                                      feed_dict = {images: batch_images, 
                                                   z: batch_z, 
                                                   y: batch_labels})
            writer.add_summary(summary_str, idx+1)

            # update generator parameters.
            _, summary_str = sess.run([g_optim, g_sum], 
                                      feed_dict = {z: batch_z, 
                                                   y: batch_labels})
            writer.add_summary(summary_str, idx+1)

            # update generator parameters.
            _, summary_str = sess.run([g_optim, g_sum], 
                                      feed_dict = {z: batch_z,
                                                   y: batch_labels})
            writer.add_summary(summary_str, idx+1)
            
            # calculate loss and print.
            errD_fake = d_loss_fake.eval({z: batch_z, y: batch_labels})
            errD_real = d_loss_real.eval({images: batch_images, y: batch_labels})
            errG = g_loss.eval({z: batch_z, y: batch_labels})

            if idx % 20 == 0:
                print("Epoch: [%2d] [%4d/%4d] d_loss: %.8f, g_loss: %.8f" \
                        % (epoch, idx, batch_idxs, errD_fake+errD_real, errG))
            
            # sample images and save to picture.
            # /home/your_name/TensorFlow/DCGAN/samples/
            if idx % 100 == 1:
                sample = sess.run(samples, feed_dict = {z: sample_z, y: sample_labels})
                samples_path = '/Users/yy/Documents/github/GAN-tensorflow/samples/'
                save_images(sample, [8, 8], 
                            samples_path + 'test_%d_epoch_%d.png' % (epoch, idx))
                print 'save down'
            
            # every 500 epochs save model.
            if idx % 500 == 2:
                checkpoint_path = os.path.join(train_dir, 'DCGAN_model.ckpt')
                saver.save(sess, checkpoint_path, global_step = idx+1)
                
    sess.close()



if __name__ == '__main__':
    train()
