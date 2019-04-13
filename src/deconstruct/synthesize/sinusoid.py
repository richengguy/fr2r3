import numpy as np
from scipy.signal.windows import kaiser


def sinusoid(size, frequency, angle, beta=0):
    '''Generate an image with pure sinusoid.

    The amplitude and DC offset are set so that the minimum and maximum values
    are 0 and 1, respectively.

    Parameters
    ----------
    size : tuple
        width and height of the output image
    frequency : float
        frequency in cycles per pixel
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
    out = np.cos(omega*u) / 2.0 + 0.5

    w1 = kaiser(size[0], beta)
    w2 = kaiser(size[1], beta)

    window = w1[np.newaxis, :].T @ w2[np.newaxis, :]
    out *= window

    return out
