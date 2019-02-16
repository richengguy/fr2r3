import numpy as np

import skimage.filters


def corner(img, sigma=1.0):
    '''Compute the structure tensor of an image.

    The structure tensor is computed independently for each channel and the
    results are stacked together.  Some manipulation is done to make the result
    look good (for some definition of "look good").

    Parameters
    ----------
    img : numpy.ndarray
        image array
    sigma : float
        smoothing amount of the Gaussian pre-filter applied onto the image

    Returns
    -------
    numpy.ndarray
        the smallest eigenvalue of the structure tensor (highlights corners)
    '''
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    R = np.zeros_like(img, dtype=np.float)

    for i in range(img.shape[2]):
        gx = skimage.filters.sobel_v(img[:, :, i])
        gy = skimage.filters.sobel_h(img[:, :, i])

        # Defines the three unique values in the structure tensor, S.
        Ixx = skimage.filters.gaussian(gx * gx, sigma=sigma)
        Ixy = skimage.filters.gaussian(gx * gy, sigma=sigma)
        Iyy = skimage.filters.gaussian(gy * gy, sigma=sigma)

        # Compute the corner response from the Eigenvalues.
        R[:, :, i] = ((Ixx + Iyy) - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))/2

    minval = 1e6
    maxval = -minval
    for i in range(img.shape[2]):
        minval = min(minval, np.min(R[:, :, i]))
        maxval = max(maxval, np.max(R[:, :, i]))

    R = (R - minval) / (maxval - minval)
    R[R < 0] = 0
    R[R > 1] = 1

    return R
