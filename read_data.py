import os
import numpy as np

"""
('Concat trX ', (60000, 28, 28, 1), ' and teX ', (10000, 28, 28, 1), ' on axis 0 ')
('Concat trY ', (60000,), ' and teY ', (10000,), ' on axis 0 ')
('dataX size is ', (70000, 28, 28, 1), ' and data y is ', (70000, 10))
"""
def read_data():

    data_dir = '/Users/yy/Documents/github/GAN-tensorflow/data/mnist'

    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    # transform to array.
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    # from the sixteen char
    trX = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

    # train label.
    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    # test data.
    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

    # test label.
    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)

    # G network follow the noise distribution. no need to use test data. Combine traing set and test set.
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0)
    print("Concat trX ",trX.shape," and teX ", teX.shape, " on axis 0 ")
    print("Concat trY ",trY.shape," and teY ", teY.shape, " on axis 0 ")

    # shuffle.
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    # y_vec is retruction. here it's the label one-hot encode.
    y_vec = np.zeros((len(y), 10), dtype=np.float)
    for i, label in enumerate(y):
        y_vec[i, int(y[i])] = 1.0

    return X / 255., y_vec
