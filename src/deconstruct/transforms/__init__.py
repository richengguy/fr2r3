from deconstruct.transforms.fourier import fourier
from deconstruct.transforms.corner import corner
from deconstruct.transforms.hough import hough
from deconstruct.transforms.wavelet import wavelet

# List of all available transforms.
TRANSFORMS = {
    'fourier': fourier,
    'corner': corner,
    'hough': hough,
    'wavelet': wavelet
}
