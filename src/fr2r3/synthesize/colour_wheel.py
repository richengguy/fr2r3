import numpy as np
import skimage


def colour_wheel(size):
    '''Generate a colour wheel image.

    Parameters
    ----------
    size : tuple
        width and height of the output image

    Returns
    -------
    numpy.ndarray
        the generated output image
    '''
    width, height = size
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))

    # Generate the saturation.
    mag = x**2 + y**2
    sat = np.sqrt(mag)

    # The hue also needs to be adjust since atan2() returns a value between
    # -pi and pi, but the hue needs to be between 0 an 1.
    hue = np.arctan2(y, x)
    hue[hue < 0] = hue[hue < 0] + 2*np.pi
    hue /= 2*np.pi

    # Generate the value by setting it as a disk.
    val = np.ones_like(mag)
    val[np.sqrt(mag) < 0.5] = 0
    val[np.sqrt(mag) > 1.0] = 0

    output = np.dstack((hue, sat, val))
    output = skimage.color.hsv2rgb(output)

    return output
