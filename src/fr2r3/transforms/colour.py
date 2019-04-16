import numpy as np

import skimage.color
import skimage.filters

from sklearn.mixture import BayesianGaussianMixture


def generate_samples(width, height, samples):
    '''Generate the set of image positions.

    Parameters
    ----------
    width, height : int
        image dimensions
    samples : int
        number of samples to obtain from the image

    Returns
    -------
    numpy.ndarray
        2xN array containing the sample positions
    '''
    num_pixels = width*height
    indices = set(np.random.randint(num_pixels, size=samples))
    while len(indices) < samples:
        indices.add(np.random.randint(num_pixels, size=1)[0])

    y, x = np.unravel_index(list(indices), (height, width))

    return np.transpose(np.c_[x, y])


def colour(img, scale=1.0, samples=10000):
    '''Model the distribution of colours in an image.

    The method models the distribution of the values in the chroma channels of
    an image after being converted from RGB to CIE Lab.  This decouples the
    luma (intensity) values from the chroma (colour) information, make it
    easier to visualize how the colours themselves appear.  The resulting
    visualization is the same size as the original image.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    scale : float
        image scaling factor
    samples : int
        number of samples to draw when generating the density estimate

    Returns
    -------
    numpy.ndarray
        a new image, same dimensions as the input, visualizing the colour
        distribution

    Raises
    ------
    ValueError
        if the input image is not an RGB image
    '''
    if img.ndim != 3:
        raise ValueError('Require RGB image to compute structure tensor.')

    img = skimage.transform.rescale(img, 1.0/scale, anti_aliasing=True,
                                    mode='constant', multichannel=True)
    img = skimage.color.rgb2hsv(img)
    height, width = img.shape[0:2]

    # Extract the colour vectors and sample from them.
    ind = generate_samples(width, height, samples)
    X = np.squeeze(img[ind[1, :], ind[0, :], 0:2])

    # Convert a polar to cartesian coordinate conversation (will make the
    # visualization easier).
    mag = X[:, 1]
    ang = 2*np.pi*X[:, 0]

    X[:, 0] = mag*np.cos(ang)
    X[:, 1] = mag*np.sin(ang)

    # Perform a density estimation using a GMM.
    gmm = BayesianGaussianMixture(
        n_components=25,
        weight_concentration_prior_type='dirichlet_distribution',
        weight_concentration_prior=1e-3)
    gmm.fit(X)

    # Generate the output array.
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    X = np.c_[x.flatten(), y.flatten()]
    scores = np.exp(gmm.score_samples(X))
    max_score = np.max(scores)

    # Apply a gamma correction to make the image look a bit nicer.
    val = np.reshape(scores, (height, width)) / max_score
    val = skimage.exposure.adjust_gamma(val, gamma=0.3)

    # Convert back from HSV to RGB.  The saturation needs to be clamped so that
    # it doesn't produce invalid values during the HSV->RGB conversion.
    mag = x**2 + y**2
    sat = np.sqrt(mag)
    sat[sat > 1] = 1

    # The hue also needs to be adjust since atan2() returns a value between
    # -pi and pi, but the hue needs to be between 0 an 1.
    hue = np.arctan2(y, x)
    hue[hue < 0] = hue[hue < 0] + 2*np.pi
    hue /= 2*np.pi

    output = np.dstack((hue, sat, val))
    output = skimage.color.hsv2rgb(output)
    output = skimage.transform.rescale(output, scale, anti_aliasing=True,
                                       mode='constant', multichannel=True)

    return output
