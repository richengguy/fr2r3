import numpy as np

import skimage.filters


def corner(img, scale=1.0):
    '''Compute the structure tensor of an image.

    The structure tensor is computed independently for each channel and the
    results are stacked together.  Some manipulation is done to make the result
    look good (for some definition of "look good").

    Parameters
    ----------
    img : numpy.ndarray
        image array
    scale : float
        smoothing amount of the Gaussian pre-filter applied onto the image

    Returns
    -------
    numpy.ndarray
        the smallest eigenvalue of the structure tensor (highlights corners)
    '''
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    img = skimage.transform.rescale(img, 1.0/2.0, anti_aliasing=True,
                                    mode='constant', multichannel=True)

    R = np.zeros_like(img, dtype=np.float)

    for i in range(img.shape[2]):
        gx = skimage.filters.sobel_v(img[:, :, i])
        gy = skimage.filters.sobel_h(img[:, :, i])

        # Defines the three unique values in the structure tensor, S.
        Ixx = skimage.filters.gaussian(gx * gx, sigma=scale)
        Ixy = skimage.filters.gaussian(gx * gy, sigma=scale)
        Iyy = skimage.filters.gaussian(gy * gy, sigma=scale)

        # Compute the corner response from the Eigenvalues.
        R[:, :, i] = ((Ixx + Iyy) - np.sqrt((Ixx - Iyy)**2 + 4*Ixy**2))/2

    minval = 1e6
    maxval = -minval
    for i in range(img.shape[2]):
        minval = min(minval, np.min(R[:, :, i]))
        maxval = max(maxval, np.max(R[:, :, i]))

    maxval *= 0.5
    R = (R - minval) / (maxval - minval)
    R[R < 0] = 0
    R[R > 1] = 1

    R = skimage.transform.rescale(R, 2.0, anti_aliasing=True,
                                  mode='constant', multichannel=True)

    return R
