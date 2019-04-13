import matplotlib.cm as cm

import click

import numpy as np

import skimage.color
import skimage.feature
import skimage.morphology
import skimage.transform

__all__ = [
    'hough'
]


def _edge_map(gx, gy):
    '''Generate an HSV-coded edge map.

    Parameters
    ----------
    gx, gy : numpy.ndarray
        image gradients

    Returns
    -------
    numpy.ndarray
        RGB image containing the edge map
    '''
    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx) + np.pi

    mag = mag / np.max(mag)
    ang = ang / (2*np.pi)

    out = np.zeros((gx.shape[0], gx.shape[1], 3))
    out[:, :, 0] = ang
    out[:, :, 1] = mag
    out[:, :, 2] = mag

    return skimage.color.hsv2rgb(out)


def hough(img, scale=1.0):
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
        value of the Gaussian filter's "sigma"

    Returns
    -------
    hough : numpy.ndarray
        visualization of the Hough parameter space
    edges : numpy.ndarray
        edges used to generate for the SHT
    '''
    click.secho('WARNING: ', fg='yellow', nl=False)
    click.echo('The Hough implementation is not optimized; this will be very slow.')  # noqa: E501

    grey = skimage.color.rgb2gray(img)
    grey = skimage.filters.gaussian(grey, scale)

    # Compute the gradients.
    gx = skimage.filters.sobel_h(grey)
    gy = skimage.filters.sobel_v(grey)

    mag = np.sqrt(gx**2 + gy**2)
    ang = np.arctan2(gy, gx)

    # For each pixel, figure out where it goes into a Hough accumulator,
    # weighting it by the magnitude.   This is a slightly modified version of
    # Algorithm 4.2 in "Computer Vision: Algorithm and Applications" by Richard
    # Szeliski.
    x, y = np.meshgrid(np.arange(mag.shape[1]), np.arange(mag.shape[0]))

    # Compute the line scalar parameters, given the normals.
    d = np.cos(ang)*x + np.sin(ang)*y
    d_max = np.sqrt(d.shape[0]**2 + d.shape[1]**2)

    # Scale everything to be between 0 and 1.
    d = (d + d_max) / (2*d_max)
    ang = (ang + np.pi) / (2*np.pi)
    mag = mag / np.max(mag)

    i = d*(mag.shape[0]-1)
    j = ang*(mag.shape[1]-1)

    # Precompute the indices so there's less work going on in the for-loop.
    ii = np.zeros((i.shape[0], i.shape[1], 2))
    jj = np.zeros((j.shape[0], j.shape[1], 2))

    ii[:, :, 0] = np.floor(i)
    ii[:, :, 1] = np.ceil(i)

    jj[:, :, 0] = np.floor(j)
    jj[:, :, 1] = np.ceil(j)

    ii = ii.astype(np.int)
    jj = jj.astype(np.int)

    # Now, do the actual Hough transform.
    hough = np.zeros_like(d)
    for y in range(hough.shape[0]):
        for x in range(hough.shape[1]):
            # The accumulator pixel doesn't fall exactly into one bin, so some
            # smoothing is done by adding to the four possible bins.
            i1 = ii[y, x, 0]
            i2 = ii[y, x, 1]

            j1 = jj[y, x, 0]
            j2 = jj[y, x, 1]

            hough[i1, j1] += mag[y, x]
            hough[i1, j2] += mag[y, x]
            hough[i2, j1] += mag[y, x]
            hough[i2, j2] += mag[y, x]

    # Apply some histogram-based thresholding to make the accumulators a bit
    # easier to see.
    hough = hough / hough.max()
    hist, bins = np.histogram(hough[:], bins=256, range=(0, 1))
    for i in reversed(range(256)):
        if hist[i] > 25:
            th = bins[i+1]
            break

    hough /= th
    hough[hough > 1] = 1

    out = cm.afmhot(hough)
    out = skimage.color.rgba2rgb(out)

    return out, _edge_map(gx, gy)
