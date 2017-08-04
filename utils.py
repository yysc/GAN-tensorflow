import scipy.misc
import numpy as np

# Save images.
def save_images(images, size, path):
    
    """
    Save the samples images
    The best size number is
            int(max(sqrt(image.shape[0]),sqrt(image.shape[1]))) + 1
    example:
        The batch_size is 64, then the size is recommended [8, 8]
        The batch_size is 32, then the size is recommended [6, 6]
    """

    # images to tanh
    img = (images + 1.0) / 2.0
    h, w = img.shape[1], img.shape[2]

    # save batch_size images.
    merge_img = np.zeros((h * size[0], w * size[1], 3))

    # save image to target position, part of the picture. 
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        merge_img[j*h:j*h+h, i*w:i*w+w, :] = image
    
    # save picture.
    return scipy.misc.imsave(path, merge_img)