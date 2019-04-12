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


def colour(img, scale=1.0, samples=50000):
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
    img = skimage.color.rgb2yuv(img)
    height, width = img.shape[0:2]

    # Extract the colour vectors and sample from them.
    ind = generate_samples(width, height, samples)
    X = np.squeeze(img[ind[1, :], ind[0, :], 1:3])

    # Perform a density estimation using a GMM.
    gmm = BayesianGaussianMixture(
        n_components=25,
        weight_concentration_prior_type='dirichlet_distribution',
        weight_concentration_prior=1e-3)
    # gmm = KernelDensity(bandwidth=0.5)
    gmm.fit(X)

    # Generate the output array.
    u, v = np.meshgrid(np.linspace(-0.5, 0.5, width),
                       np.linspace(-0.5, 0.5, height))
    x = np.c_[u.flatten(), v.flatten()]
    scores = np.exp(gmm.score_samples(x))
    max_score = np.max(scores)

    # Apply a gamma correction to make the image look a bit nicer.
    Y = np.reshape(scores, (height, width))
    Y = (Y / max_score)**0.4

    # Convert back from Yuv to RGB.
    output = np.dstack((Y, Y*u, Y*v))
    output = skimage.color.yuv2rgb(output)
    output = skimage.transform.rescale(output, scale, anti_aliasing=True,
                                       mode='constant', multichannel=True)

    return output
