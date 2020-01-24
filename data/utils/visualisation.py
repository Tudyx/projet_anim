import numpy as np
import matplotlib.pyplot as plt


def visualise_image_samples(images, labels=None, n_images=9):
    assert np.sqrt(n_images) - np.sqrt(n_images) < 1e-12, "Expected n_images to be a square number," \
                                                          " but got {} instead".format(n_images)

    n = int(np.sqrt(n_images))
    fig, axes = plt.subplots(n, n)
    _images = images[:n_images]

    for ax, image, label in zip(axes.flatten(), images, labels):
        ax.imshow(image)
        ax.set_xticks([])
        ax.set_yticks([])
        if labels is not None:
            ax.set_title("Class: {}".format(label))
