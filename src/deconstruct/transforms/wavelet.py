import numpy as np
import pywt


def wavelet(img, nlevels=3, wavelet='bior3.9'):
    '''Apply the DWT to an image.

    This does exactly what it says: applies a the Discrete Wavelet Transform
    onto an image.  Each channel is decomposed separately, with some
    stylization to make it look a bit nicer.

    Parameters
    ----------
    img : numpy.ndarray
        input image
    nlevels : int
        number of levels in the DWT decomposition
    wavelet : str
        the wavelet type; must be one of the PyWavelet supported types

    Returns
    -------
    numpy.ndarray
        stylized wavelet decomposition
    '''
    if img.ndim == 2:
        img = img[:, :, np.newaxis]

    wavelet = pywt.Wavelet(wavelet)

    # Perform a per-channel DWT.
    channel_coeffs = []
    for i in range(img.shape[2]):
        coeffs = pywt.wavedec2(img[:, :, i], wavelet, level=nlevels)

        for j in range(1, len(coeffs)):
            coeffs[j] = list(coeffs[j])

        channel_coeffs.append(coeffs)

    # Scale the approximation signals to be in the usual image range.
    minval = 1e6
    maxval = -1e6

    for coeff in channel_coeffs:
        minval = min(minval, coeff[0].min())
        maxval = max(maxval, coeff[0].max())

    for coeff in channel_coeffs:
        coeff[0] = (coeff[0] - minval) / (maxval - minval)

    # Process the detail signals to make them appear interesting.
    for i in range(1, len(channel_coeffs[0])):
        for j in range(3):
            maxval = -1e6
            for coeff in channel_coeffs:
                coeff[i][j] = np.abs(coeff[i][j])
                maxval = max(maxval, coeff[i][j].max())
            for coeff in channel_coeffs:
                coeff[i][j] /= maxval

    # Prep for output.
    out = []
    for coeff in channel_coeffs:
        channel, _ = pywt.coeffs_to_array(coeff)
        out.append(channel)

    return np.dstack(out)
