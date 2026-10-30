
import numpy as np
import matplotlib.pyplot as plt


def make_gaussian_mask2d(size, sigma):
    r = size // 2
    x = np.arange(-r, r + 1)
    y = np.arange(-r, r + 1)
    X, Y = np.meshgrid(x, y)

    gmask = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    gmask = gmask / np.sum(gmask)

    return gmask


def imgray(images):
    n = len(images)

    plt.figure(figsize=(5*n, 5))

    for i, img in enumerate(images):
        plt.subplot(1, n, i+1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255)
        plt.axis('off')

    plt.tight_layout()
    plt.show()
