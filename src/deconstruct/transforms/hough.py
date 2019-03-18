import matplotlib.cm as cm

import numpy as np

import skimage.color
import skimage.feature
import skimage.morphology
import skimage.transform

import scipy.ndimage

__all__ = [
    'hough'
]


def hough(img, sigma=1.0, scale=1.0):
    '''Generates a Hough transform visualization.

    The visualization is obtained by first running an edge detector followed
    by the Standard Hough Transform.  Since the goal is to make a nice looking
    image, not accurately find lines, any image with strong linear segments
    will be "good enough" for this purpose.

    One important thing to keep in mind is that the size of the Hough parameter
    space isn't the same as the original image.  To make this look reasonable,
    the transform is stretched so that it the same dimensions as the input
    image.  While this invalidates the transform, the purpose, again, is for
    visualization.

    Parameters
    ----------
    img : numpy.ndarray
        input RGB image
    sigma : float
        smoothing amount for the Canny edge detector's Gaussian pre-filter
    scale : float
        amount to scale the image before processing

    Returns
    -------
    hough : numpy.ndarray
        visualization of the Hough parameter space
    edges : numpy.ndarray
        edges used to generate for the SHT
    '''
    img = skimage.color.rgb2gray(img)
    size = img.shape

    if scale > 1:
        img = skimage.transform.rescale(img, 1.0/scale, anti_aliasing=True,
                                        mode='constant', multichannel=False)

    edges = skimage.feature.canny(img, sigma=sigma)

    angles = np.linspace(-np.pi, np.pi, size[1])
    hough, _, _ = skimage.transform.hough_line(edges, angles)

    kh = size[0] / hough.shape[0]
    kw = size[1] / hough.shape[1]

    hough = scipy.ndimage.zoom(hough, (kh, kw))
    hough = hough / hough.max()

    hist, bins = np.histogram(hough[:], bins=256, range=(0, 1))

    for i in reversed(range(256)):
        if hist[i] > 25:
            maxval = bins[i+1]
            break

    hough /= maxval
    hough[hough > 1] = 1

    outimg = cm.afmhot(hough)
    outimg = skimage.color.rgba2rgb(outimg)

    return outimg, edges
