import enum

import numpy as np
from PIL import Image


__all__ = [
    'imread',
    'imshow',
]


_EXIF_ORIENTATION = 274


class ExifOrientation(enum.IntEnum):
    '''Various orientations stored within EXIF data.

    The values used below come from the EXIF specification.  The enum names
    have been chosen to represent the equivalent clockwise rotation rather than
    the corner-corner mapping from the original specification.
    '''
    Original = 1
    Rotate90 = 8
    Rotate180 = 3
    Rotate270 = 6


def get_orientation(img):
    '''Check the EXIF data to see if the image has a preferred orientation.

    This is only possible if the image has associated EXIF data.  If there is
    no data then the image is treated as if it has a rotation flag of '0'.

    Parameters
    ----------
    img: PIL.Image
        a Pillow image object

    Returns
    -------
    ExifOrientation:
        enum representing the EXIF orientation flag
    '''
    try:
        exif = img._getexif()
    except AttributeError:
        return ExifOrientation.Original

    if _EXIF_ORIENTATION in exif:
        return ExifOrientation(exif[_EXIF_ORIENTATION])
    else:
        return ExifOrientation.Original


def imread(fname, return_exif=False, inspect_exif=True):
    '''Read in an image file.

    This provides a wrapper around the various libraries used to load image
    data.  It will always provide the final output using Pillow to keep the
    interface consistent.  Use :func:`to_ndarray` to convert the object into a
    numpy-compatible array.

    Parameters
    ----------
    fname : str
        path to the image file
    inspect_exif : bool
        look at the image EXIF data and apply any orientation changes post-load
    return_exif : bool
        return EXIF data alongside the original image

    Returns
    -------
    img : PIL.Image.Image
        the opened image
    exif : dict
        the image's EXIF data (only if ``return_exif`` is ``True``)
    '''
    img = Image.open(fname)
    exif = img._getexif()

    if exif is not None and inspect_exif:
        orientation = get_orientation(img)
        if orientation == ExifOrientation.Rotate90:
            img = img.rotate(90, Image.BICUBIC, True)
        elif orientation == ExifOrientation.Rotate180:
            img = img.rotate(180, Image.BICUBIC)
        elif orientation == ExifOrientation.Rotate270:
            img = img.rotate(270, Image.BICUBIC, True)

    if return_exif:
        return img, exif
    else:
        return img


def imshow(img):
    '''Display an image using Matplotlib.

    Parameters
    ----------
    img: numpy.ndarray or PIL.Image.Image
        the image to display
    '''
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()


def to_ndarray(img, sz=None, ignore_exif=False):
    '''Convert an image into a numpy-compatible ndarray.

    The Image class stores a representation of the image, not the image itself,
    so some work is needed to convert it into a usable format.  Furthermore, in
    the case of camera images, these may have metadata, such as EXIF,
    indicating that the image requires a rotation.

    Note
    ----
    This may consume a lot of memory depending on the image size.  Pillow
    usually stores the compressed representation so this requires making an
    in-memory copy.

    Parameters
    ----------
    img: PIL.Image
        image object being converted
    sz: (width, height)
        if provided then this will specify the maximum width and height of the
        image
    ignore_exif: bool
        ignore any image manipulations based on the EXIF data

    Returns
    -------
    numpy.ndarray
        the numpy representation
    '''
    if not isinstance(img, Image.Image):
        raise ValueError('Expected a PIL.Image instance.')

    # Extract the EXIF data if it exists.
    try:
        exif = img._getexif()
    except AttributeError:
        ignore_exif = True

    # Create a copy of the image before resizing it.
    img = img.copy()
    if sz is not None:
        img.thumbnail(sz)

    # Convert into the numpy float array.
    data = np.array(img, dtype=np.float64)
    data = data / 255.0

    # Finally, check to see if a rotation is required.
    if exif is not None and not ignore_exif:
        if _EXIF_ORIENTATION in exif:
            rotation = ExifOrientation(exif[_EXIF_ORIENTATION])
        else:
            rotation = ExifOrientation.Original

        # Rotate the image depending on the EXIF tag value.
        if rotation == ExifOrientation.Rotate180:
            data = np.flip(data, 1)
        elif rotation == ExifOrientation.Rotate90:
            data = np.swapaxes(data, 0, 1)
            data = np.flip(data, 0)
        elif rotation == ExifOrientation.Rotate270:
            raise NotImplementedError('Need to implement EXIF Orientation 6')

    return data


def to_pil_image(arr):
    '''Convert a numpy array into a Pillow image.

    Parameters
    ----------
    img : numpy.ndarray
        an image stored as a numpy array

    Returns
    -------
    PIL.Image
        image object
    '''
    if arr.dtype != np.uint8:
        arr = np.uint8(255 * arr / np.max(arr))

    return Image.fromarray(arr)
