import numpy as np


def edge(size, frequency, angle, beta=0):
    '''Generate an image with periodic edges.

    Parameters
    ----------
    size : tuple
        width and height of the output image
    frequency : float
        the frequency of the pulse train, in cycles per pixel
    angle : float
        value in degrees indicating the orientation of the sinusoid
    beta : float
        value of the Kaiser window 'beta' parameter

    Returns
    -------
    numpy.ndarray
        the generated output image
    '''
    if frequency < 0:
        raise ValueError('Frequency must be a positive value.')

    omega = 2.0 * np.pi * frequency / np.min(size)
    theta = np.deg2rad(angle)

    y, x = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    u = x * np.cos(theta) + y * np.sin(theta)
    out = np.cos(omega*u)
    out[out >= 0] = 1
    out[out < 0] = 0

    return out
