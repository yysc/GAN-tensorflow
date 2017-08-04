import os
from model import *


def train():
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_dir = './log'

    label = tf.placeholder(tf.float32, [BATCH_SIZE, 10], name='rawLabels')
    images = tf.placeholder(
        tf.float32, [BATCH_SIZE, 28, 28, 1], name='realImages')
    noise = tf.placeholder(tf.float32, [BATCH_SIZE, 100], name='randomNoise')

    G = generator(noise, label)


if __name__ == "__main__":
    train()
