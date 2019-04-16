import skimage.color
import numpy as np

__all__ = [
    'fourier'
]


def _compute_dft(img):
    '''Compute the DFT of an image.

    Parameters
    ----------
    img : numpy.ndarray
        input image

    Returns
    -------
    mag : numpy.ndarray
        DFT magnitude
    '''
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)

    re = np.real(dft)
    im = np.imag(dft)

    mag = np.sqrt(re**2 + im**2)

    return mag


def fourier(img, scale=1):
    '''Generate a stylized Fourier transform.

    The transform visualizes the DFT by encoding the image's log-magnitude PSD
    as value and the phase as hue.  To keep things simple, the image is first
    converted into greyscale.

    Parameters
    ----------
    img : numpy.ndarray
        image transform is applied to
    scale : int
        scaling factor applied to the image

    Returns
    -------
    numpy.ndarray
        image's DFT as an RGB image
    '''
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    if scale > 1:
        scaled_img = skimage.transform.rescale(img, 1.0/scale,
                                               anti_aliasing=True,
                                               mode='constant',
                                               multichannel=True)
        h, w = scaled_img.shape[:2]
        img = np.zeros_like(img)
        img[:h, :w, :] = scaled_img

    channels = []
    for i in range(img.shape[2]):
        mag = _compute_dft(img[:, :, i])
        psd = mag**2

        logpsd = np.log10(1.0 + psd)
        logpsd = (logpsd - np.min(logpsd)) / (np.max(logpsd) - np.min(logpsd))

        channels.append(logpsd)

    out = np.dstack(channels)
    out = np.squeeze(out)

    if scale > 1:
        return out, img
    else:
        return out
