import numpy as np

import skimage.filters
import skimage.color


def x_derivative(img):
    '''Compute the RGB x-derivative of an image.

    Parameters
    ----------
    img : numpy.ndarray
        input RGB image

    Returns
    -------
    dR : numpy.ndarray
        derivative of the red channel
    dG : numpy.ndarray
        derivative of the green channel
    dB : numpy.ndarray
        derivative of the blue channel
    '''
    dR = skimage.filters.sobel_h(img[:, :, 0])
    dG = skimage.filters.sobel_h(img[:, :, 1])
    dB = skimage.filters.sobel_h(img[:, :, 2])
    return dR, dG, dB


def y_derivative(img):
    '''Compute the RGB y-derivative of an image.

    Parameters
    ----------
    img : numpy.ndarray
        input RGB image

    Returns
    -------
    dR : numpy.ndarray
        derivative of the red channel
    dG : numpy.ndarray
        derivative of the green channel
    dB : numpy.ndarray
        derivative of the blue channel
    '''
    dR = skimage.filters.sobel_v(img[:, :, 0])
    dG = skimage.filters.sobel_v(img[:, :, 1])
    dB = skimage.filters.sobel_v(img[:, :, 2])
    return dR, dG, dB


def dotprod(a, b):
    '''Convenience method to compute vector dot-product.

    Parameters
    ----------
    a, b : tuple(R, G, B)
        the output tuple from one of the two derivative methods

    Returns
    -------
    numpy.ndarray
        dot-product between the two tuples
    '''
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]


def compute(img, scale=1.0):
    '''Compute the structure tensor of an image.

    The structure tensor is computed using the RGB variant described in
    [kyprianidis2008]_.  As such, it requires that the input image has three
    colour channels.  An exeception is raised if this is not the case.

    .. [kyprianidis2008] Kyprianidis, J. E., & Döllner, J. (2008). Image
       Abstraction by Structure Adaptive Filtering. In: Proc. EG UK Theory and
       Practice of Computer Graphics, pp. 51–58.

    Parameters
    ----------
    img : numpy.ndarray
        image array
    scale : float
        smoothing amount of the Gaussian pre-filter applied onto the image

    Returns
    -------
    numpy.ndarray
        an RGB image that visualizes the eigenvectors within the structure
        tensor

    Raises
    ------
    ValueError
        if the input image is not an RGB image
    '''
    if img.ndim == 2:
        raise ValueError('Require RGB image to compute structure tensor.')

    # Compute the derivatives.
    dfdx = x_derivative(img)
    dfdy = y_derivative(img)

    # Compute the components of the structure tensor.
    E = dotprod(dfdx, dfdx)
    F = dotprod(dfdx, dfdy)
    G = dotprod(dfdy, dfdy)

    # Apply a gaussian filter to the components to smooth out the tensor.
    E = skimage.filters.gaussian(E, sigma=scale)
    F = skimage.filters.gaussian(F, sigma=scale)
    G = skimage.filters.gaussian(G, sigma=scale)

    return E, F, G


def eigenvectors(E, F, G):
    '''Compute the eigenvectors of the structure tensor.

    The method used to obtain the eigenvectors is described in [cumani1989]_.

    .. [cumani1989] Cumani, A. (1991). Edge Detection in Multispectral Images.
       In: Graphical Models and Image Processing, pp. 40-51

    Parameters
    ----------
    E, F, G : numpy.ndarray
        the output of :meth:`compute`

    Returns
    -------
    v1 : numpy.ndarry
        a two-channel image containing the eigenvector of the major eigenvalue;
        corresponds to edge normals
    v2 : numpy.ndarray
        a two-channel image containing the eigenvector of the minor eigenvalue;
        corresponds to edge tangents
    '''
    EG = E + G
    det = np.sqrt((E - G)**2 + 4*F**2)

    l1 = (EG + det) / 2
    l2 = (EG - det) / 2

    theta = 0.5*np.arctan(2*F / (E - G))

    v1 = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    v2 = np.stack((-v1[:, :, 1], v1[:, :, 0]), axis=-1)

    return l1[:, :, np.newaxis]*v1, l2[:, :, np.newaxis]*v2


def visualize(img, scale=1.0):
    '''Visualize the structure tensor.

    The visualization uses the components of the structure tensor to set the
    saturation and value of each pixel in the output image.  It is set up so
    that long, linear segments appear white while corners are coloured.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    scale : float
        smoothing amount of the Gaussian pre-filter applied onto the image

    Returns
    -------
    numpy.ndarray
        an RGB image that visualizes the eigenvectors within the structure
        tensor
    '''
    img = skimage.transform.rescale(img, 1.0/2.0, anti_aliasing=True,
                                    mode='constant', multichannel=True)

    E, F, G = compute(img, scale)

    # The visualization is quite simple.  The hue is fixed while the saturation
    # depends on 'F' while the value depends on 'E + G'.
    R = np.zeros_like(img, dtype=np.float)
    R[:, :, 0] = 180
    R[:, :, 1] = F/np.max(F)
    R[:, :, 2] = (E + G)/np.max(E + G)

    R = skimage.color.hsv2rgb(R)
    R = skimage.transform.rescale(R, 2.0, anti_aliasing=True,
                                  mode='constant', multichannel=True)

    return R
